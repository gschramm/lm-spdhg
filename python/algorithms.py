import numpy as np
from pyparallelproj.models import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm
from pyparallelproj.utils import GradientOperator, GradientNorm
from utils import count_event_multiplicity

def spdhg_lm(events, attn_list, sens_list, contam_list, sens_img,
             proj, niter, nsubsets, 
             x0 = None, fwhm = 0, gamma = 1., rho = 0.999, rho_grad = 0.999, verbose = False, 
             callback = None, subset_callback = None,
             callback_kwargs = None, subset_callback_kwargs = None,
             grad_norm = None, grad_operator = None, beta = 0, Tmode = 0):

  # count the "multiplicity" of every event in the list
  # if an event occurs n times in the list of events, the multiplicity is n
  mu = count_event_multiplicity(events, use_gpu_if_possible = True)
 
  img_shape = tuple(proj.img_dim)

  if grad_operator is None:
    grad_operator = GradientOperator()

  if grad_norm is None:
    grad_norm = GradientNorm()

  # setup the probabilities for doing a pet data or gradient update
  # p_g is the probablility for doing a gradient update
  # p_p is the probablility for doing a PET data subset update
 
  if beta == 0:
    p_g = 0
  else: 
    p_g = 0.5
    # norm of the gradient operator = sqrt(ndim*4)
    ndim  = len([x for x in img_shape if x > 1])
    grad_op_norm = np.sqrt(ndim*4)
  
  p_p = (1 - p_g) / nsubsets
  
  #--------------------------------------------------------------------------------------------
  # initialize variables
  if x0 is None:
    x = np.zeros(img_shape, dtype = np.float32)
  else:
    x = x0.copy()

  # initialize y for data
  y = 1 - (mu / (pet_fwd_model_lm(x, proj, events, attn_list, sens_list, fwhm = fwhm) + contam_list))

  z = sens_img + pet_back_model_lm((y - 1)/mu, proj, events, attn_list, sens_list, fwhm = fwhm)
  zbar = z.copy()

  # allocate arrays for gradient operations
  y_grad = np.zeros((len(img_shape),) + img_shape, dtype = np.float32)

  # calculate S for the gradient operator
  if p_g > 0:
    S_g = gamma*rho_grad/grad_op_norm
    T_g = p_g*rho_grad/(gamma*grad_op_norm)

  # calculate the "step sizes" S_i for the PET fwd operator
  S_i = []

  ones_img = np.ones(img_shape, dtype = np.float32)

  for i in range(nsubsets):
    ss = slice(i,None,nsubsets)
    # clip inf values
    tmp = pet_fwd_model_lm(ones_img, proj, events[ss,:], attn_list[ss], sens_list[ss], fwhm = fwhm)
    tmp = np.clip(tmp, tmp[tmp > 0].min(), None)
    S_i.append(gamma*rho/tmp)

  # calculate the step size T
  # sens img is back_model(1)

  if Tmode == 0:
    tmp = sens_img / nsubsets
    T = np.divide(p_p*rho, gamma*tmp, out = np.zeros_like(tmp), where = (tmp > 0))
  else:
    Tk = np.zeros((nsubsets,) + img_shape)

    # back projection of ones along all LORs where we have data
    tmp_data = pet_back_model_lm(1/mu, proj, events, attn_list, sens_list, fwhm = fwhm)

    # back projection of ones along all LORS where we don't have data
    tmp_empty = sens_img - tmp_data

    for i in range(nsubsets):
      ss = slice(i,None,nsubsets)
      tmp = pet_back_model_lm(1/mu[ss], proj, events[ss,:], attn_list[ss], sens_list[ss], fwhm = fwhm) + (tmp_empty/nsubsets)
      Tk[i,...] = p_p*rho / (gamma*tmp)
    T = Tk.min(axis = 0)

  if p_g > 0:
    T = np.clip(T, None, T_g)

  #--------------------------------------------------------------------------------------------
  # SPDHG iterations

  zero_sens_inds = np.where(sens_img == 0)
  
  for it in range(niter):
    subset_sequence = np.random.permutation(np.arange(int(nsubsets/(1-p_g))))
  
    for iss in range(subset_sequence.shape[0]):
      x = np.clip(x - T*zbar, 0, None)

      # select a random subset
      i = subset_sequence[iss]

      x[zero_sens_inds] = 0
      zbar[zero_sens_inds] = 0
      z[zero_sens_inds] = 0

      if i < nsubsets:
        # PET subset update
        print(f'iteration {it + 1} step {iss} subset {i+1}')
  
        ss = slice(i,None,nsubsets)
  
        y_plus = y[ss] + S_i[i]*(pet_fwd_model_lm(x, proj, events[ss,:], attn_list[ss], sens_list[ss], 
                                                  fwhm = fwhm) + contam_list[ss])
  
        # apply the prox for the dual of the poisson logL
        y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]*mu[ss]))
  
        dz = pet_back_model_lm((y_plus - y[ss])/mu[ss], proj, events[ss,:], 
                               attn_list[ss], sens_list[ss], fwhm = fwhm)
  
        # update variables
        z = z + dz
        y[ss] = y_plus.copy()
        zbar = z + dz/p_p
      else:
        print(f'iteration {it + 1} step {iss} gradient update')
        y_grad_plus = (y_grad + S_g*grad_operator.fwd(x))

        # apply the prox for the gradient norm
        y_grad_plus = beta*grad_norm.prox_convex_dual(y_grad_plus/beta, sigma = S_g/beta)

        dz = grad_operator.adjoint(y_grad_plus - y_grad)
        dz[zero_sens_inds] = 0
  
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
