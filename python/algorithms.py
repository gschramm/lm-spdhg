import numpy as np
from pyparallelproj.models import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm
from pyparallelproj.utils import GradientOperator

def spdhg_lm(events, multi_index,
             em_sino, attn_sino, sens_sino, contam_sino, 
             proj, lmproj, niter, nsubsets,
             fwhm = 0, gamma = 1., rho = 0.999, verbose = False, 
             callback = None, subset_callback = None,
             callback_kwargs = None, subset_callback_kwargs = None,
             beta = 0, grad_operator = None):
 
  # generate the 1d contamination, sensitivity and attenuation lists from the sinograms
  attn_list   = attn_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
  sens_list   = sens_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
  contam_list = contam_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]]

  img_shape = tuple(lmproj.img_dim)

  if grad_operator is None:
    grad_operator = GradientOperator()

  # setup the probabilities for doing a pet data or gradient update
  # p_g is the probablility for doing a gradient update
  # p_p is the probablility for doing a PET data subset update
  
  if beta == 0:
    p_g = 0
  else: 
    p_g = 0.5
    # norm of the gradient operator = sqrt(ndim*4)
    ndim  = len([x for x in img_shape if x > 1])
    grad_norm = np.sqrt(ndim*4)
  
  p_p = (1 - p_g) / nsubsets
  
  #--------------------------------------------------------------------------------------------
  # initialize variables
  x = np.zeros(img_shape, dtype = np.float32)
  y = np.zeros(events.shape[0], dtype = np.float32)
  
  z  = pet_back_model((em_sino == 0).astype(np.float32), proj, attn_sino, sens_sino, 0, fwhm = fwhm)
  zbar = z.copy()
  
  # allocate arrays for gradient operations
  #x_grad      = np.zeros((len(img_shape),) + img_shape, dtype = np.float32)
  y_grad      = np.zeros((len(img_shape),) + img_shape, dtype = np.float32)
  #y_grad_plus = np.zeros((len(img_shape),) + img_shape, dtype = np.float32)

  # calculate S for the gradient operator
  if p_g > 0:
    S_g = (gamma*rho/grad_norm)
    T_g = p_g*rho/(gamma*grad_norm)

  # calculate the "step sizes" S_i for the PET fwd operator
  S_i = []

  ones_img = np.ones(img_shape, dtype = np.float32)

  for i in range(nsubsets):
    ss = slice(i,None,nsubsets)
    # clip inf values
    tmp = (gamma*rho) / pet_fwd_model_lm(ones_img, lmproj, events[ss,:5], 
                                         attn_list[ss], sens_list[ss], fwhm = fwhm)
    tmp[tmp == np.inf] = tmp[tmp != np.inf].max()
    S_i.append(tmp)


  T_i = np.zeros((nsubsets,) + img_shape, dtype = np.float32)

  for i in range(nsubsets):
    ss = slice(i,None,nsubsets)

    tmp = pet_back_model_lm(1./events[ss,5], lmproj, events[ss,:5], 
                            attn_list[ss], sens_list[ss], fwhm = fwhm) + z/nsubsets

    T_i[i,...] = p_p*rho/(gamma*tmp)

  # take the element-wise min of the T_i's of all subsets
  T = T_i.min(axis = 0)

  if p_g > 0:
    T = np.clip(T, None, T_g)

    
  #--------------------------------------------------------------------------------------------
  # SPDHG iterations
  
  for it in range(niter):
    subset_sequence = np.random.permutation(np.arange(int(nsubsets/(1-p_g))))
  
    for iss in range(subset_sequence.shape[0]):
      
      # select a random subset
      i = subset_sequence[iss]
  
      if i < nsubsets:
        # PET subset update
        print(f'iteration {it + 1} step {iss} subset {i+1}')
  
        ss = slice(i,None,nsubsets)
  
        x = np.clip(x - T*zbar, 0, None)
  
        y_plus = y[ss] + S_i[i]*(pet_fwd_model_lm(x, lmproj, events[ss,:5], 
                                                  attn_list[ss], sens_list[ss], 
                                                  fwhm = fwhm) + contam_list[ss])
  
        # apply the prox for the dual of the poisson logL
        y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]*events[ss,5]))
  
        dz = pet_back_model_lm((y_plus - y[ss])/events[ss,5], lmproj, events[ss,:5], 
                               attn_list[ss], sens_list[ss], fwhm = fwhm)
  
        # update variables
        z = z + dz
        y[ss] = y_plus.copy()
        zbar = z + dz/p_p
      else:
        print(f'iteration {it + 1} step {iss} gradient update')
  
        x_grad = grad_operator.fwd(x)
        y_grad_plus = (y_grad + S_g*x_grad).reshape(len(img_shape),-1)
  
        # proximity operator for dual of TV
        gnorm = np.linalg.norm(y_grad_plus, axis = 0)
        y_grad_plus /= np.maximum(np.ones(gnorm.shape, np.float32), gnorm / beta)
        y_grad_plus = y_grad_plus.reshape(x_grad.shape)
 
        dz = grad_operator.adjoint(y_grad_plus - y_grad)
  
        # update variables
        z = z + dz
        y_grad = y_grad_plus.copy()
        zbar = z + dz/p_g

      if subset_callback is not None:
        subset_callback(x, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)

    if callback is not None:
      callback(x, iteration = (it+1), subset = (i+1), **callback_kwargs)

  return x
