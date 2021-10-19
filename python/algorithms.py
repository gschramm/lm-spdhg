import numpy as np
from pyparallelproj.models import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm
from pyparallelproj.utils import GradientOperator
from utils import count_event_multiplicity

def spdhg_lm(events, multi_index, attn_sino, sens_sino, contam_sino, 
             proj, lmproj, niter, nsubsets,
             fwhm = 0, gamma = 1., rho = 0.999, verbose = False, 
             callback = None, subset_callback = None,
             callback_kwargs = None, subset_callback_kwargs = None,
             beta = 0, grad_operator = None):

  # count the "multiplicity" of every event in the list
  # if an event occurs n times in the list of events, the multiplicity is n
  mu = count_event_multiplicity(events, use_gpu_if_possible = True)
 
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

  # creat a tempory TOF sinogram that is 1 in all bins without counts and 0 in all other bins
  tmp = np.ones(contam_sino.shape, dtype = np.float32)
  tmp[multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]] = 0
  
  z  = pet_back_model(tmp, proj, attn_sino, sens_sino, 0, fwhm = fwhm)
  del tmp

  zbar = z.copy()
  
  # allocate arrays for gradient operations
  y_grad = np.zeros((len(img_shape),) + img_shape, dtype = np.float32)

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
    tmp = pet_fwd_model_lm(ones_img, lmproj, events[ss,:], attn_list[ss], sens_list[ss], fwhm = fwhm)
    tmp = np.clip(tmp, tmp[tmp > 0].min(), None)
    S_i.append((gamma*rho) / tmp)

  # calculate the step size T
  # the norm of every subset operator is the norm of the full operator / nsubsets
  # in theory we need to backproject a full sino of ones
  # however, z already contains a backprojections of a sino with ones in all empty data bins
  # the missing part can be done by backprojecting the LM events with a value of 1/mu 
  tmp = (pet_back_model_lm(1./mu, lmproj, events, attn_list, sens_list, fwhm = fwhm) + z)/nsubsets
  T = p_p*rho/(gamma*tmp)

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
  
        y_plus = y[ss] + S_i[i]*(pet_fwd_model_lm(x, lmproj, events[ss,:], attn_list[ss], sens_list[ss], 
                                                  fwhm = fwhm) + contam_list[ss])
  
        # apply the prox for the dual of the poisson logL
        y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]*mu[ss]))
  
        dz = pet_back_model_lm((y_plus - y[ss])/mu[ss], lmproj, events[ss,:], 
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

#------------------------------------------------------------------------------------------------

from pyparallelproj.cp_tv_denoise import cp_tv_denoise

def lm_emtv(events, attn_list, sens_list, contam_list, lmproj, sens_img, niter, nsubsets, 
            fwhm = 0, verbose = False, xstart = None, callback = None, subset_callback = None,
            callback_kwargs = None, subset_callback_kwargs = None, beta = 0, niter_denoise = 20):

  img_shape  = tuple(lmproj.img_dim)

  # initialize recon
  if xstart is None:
    recon = np.full(img_shape, events.shape[0] / np.prod(img_shape), dtype = np.float32)
  else:
    recon = xstart.copy()

  # run OSEM iterations
  for it in range(niter):
    for i in range(nsubsets):
      if verbose: print(f'iteration {it+1} subset {i+1}')

      # calculate the weights for weighted denoising problem that we have to solve
      if beta > 0:
        # post EM TV denoise step
        tmp = (recon*beta)
        tmp = np.clip(tmp, tmp[tmp > 0].min(), None)
        weights = sens_img / tmp
        # clip also max of weights to avoid float overflow
        weights = np.clip(weights, None, 0.1*np.finfo(np.float32).max)


      # EM step
      exp_list = pet_fwd_model_lm(recon, lmproj, events[i::nsubsets,:], attn_list[i::nsubsets], 
                                      sens_list[i::nsubsets], fwhm = fwhm) + contam_list[i::nsubsets]

      recon *= (pet_back_model_lm(1/exp_list, lmproj, events[i::nsubsets,:], attn_list[i::nsubsets], 
                                  sens_list[i::nsubsets], fwhm = fwhm)*nsubsets / sens_img)

      # weighted denoising
      if beta > 0:
        recon   = cp_tv_denoise(recon, weights = weights, niter = niter_denoise, nonneg = True)

      if subset_callback is not None:
        subset_callback(recon, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)

    if callback is not None:
      callback(recon, iteration = (it+1), subset = (i+1), **callback_kwargs)
      
  return recon


