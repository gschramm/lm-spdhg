#TODO: prox new operator, step sizes

# small demo for sinogram TOF OS-MLEM

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom
from pyparallelproj.models import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm
from pyparallelproj.utils import grad, div
from pyparallelproj.algorithms import spdhg

from scipy.ndimage import gaussian_filter
import numpy as np
import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 1,   type = int)
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 1,   type = int)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 4,   type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus         = args.ngpus
counts        = args.counts
niter         = args.niter
nsubsets      = args.nsubsets
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
phantom       = args.phantom
seed          = args.seed

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n0      = 120
n1      = 120
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
if phantom == 'ellipse2d':
  n   = 200
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = ellipse_phantom(n = n, c = 3)
  for i2 in range(n2):
    img[:,:,i2] = tmp
elif phantom == 'brain2d':
  n   = 128
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = brain2d_phantom(n = n)
  for i2 in range(n2):
    img[:,:,i2] = tmp

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# setup an attenuation image
att_img = (img > 0) * 0.01 * voxsize[0]

# generate nonTOF sinogram parameters and the nonTOF projector for attenuation projection
sino_params_nt = ppp.PETSinogramParameters(scanner)
proj_nt        = ppp.SinogramProjector(scanner, sino_params_nt, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus)
sino_shape_nt  = sino_params_nt.shape

attn_sino = np.zeros(sino_shape_nt, dtype = np.float32)
attn_sino = np.exp(-proj_nt.fwd_project(att_img))

# generate the sensitivity sinogram
sens_sino = np.ones(sino_shape_nt, dtype = np.float32)

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# allocate array for the subset sinogram
sino_shape = sino_params.shape
img_fwd    = np.zeros(sino_shape, dtype = np.float32)

# forward project the image
img_fwd = ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, 0, fwhm = fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)
  
  em_sino = np.random.poisson(img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with sctter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)

  em_sino = img_fwd + contam_sino

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# create a listmode projector for the LM MLEM iterations
lmproj = ppp.LMProjector(proj.scanner, proj.img_dim, voxsize = proj.voxsize, 
                         img_origin = proj.img_origin, ngpus = proj.ngpus,
                         tof = proj.tof, sigma_tof = proj.sigma_tof, tofbin_width = proj.tofbin_width,
                         n_sigmas = proj.nsigmas)

# generate list mode events and the corresponting values in the contamination and sensitivity
# for every subset sinogram

# events is a list of nsubsets containing (nevents,6) arrays where
# [:,0:2] are the transaxial/axial crystal index of the 1st detector
# [:,2:4] are the transaxial/axial crystal index of the 2nd detector
# [:,4]   are the event TOF bins
# [:,5]   are the events counts (mu_e)

events, multi_index = sino_params.sinogram_to_listmode(em_sino, 
                                                       return_multi_index = True,
                                                       return_counts = True)

contam_list = contam_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]]
sens_list   = sens_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
attn_list   = attn_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]

# backproject the subset events in LM and sino mode as a cross check
back_imgs  = np.zeros((nsubsets,) + tuple(lmproj.img_dim))

for i in range(nsubsets):
  subset_events = events[i::nsubsets,:]
  back_imgs[i,...] = lmproj.back_project(np.ones(subset_events.shape[0], dtype = np.float32), 
                                         subset_events[:,:5])

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

def calc_cost(x):
  cost = 0

  for i in range(proj.nsubsets):
    # get the slice for the current subset
    ss = proj.subset_slices[i]
    exp = ppp.pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm) + contam_sino[ss]
    cost += (exp - em_sino[ss]*np.log(exp)).sum()

  if beta > 0:
    x_grad = np.zeros((img.ndim,) + img.shape, dtype = np.float32)
    grad(x, x_grad)
    cost += beta*np.linalg.norm(x_grad, axis = 0).sum()

  return cost

def _cb(x, **kwargs):
  it = kwargs.get('iteration',0)
  it_early = kwargs.get('it_early',-1)

  if it_early == it:
    if 'x_early' in kwargs:
      kwargs['x_early'][:] = x

  if 'cost' in kwargs:
    kwargs['cost'][it-1] = calc_cost(x)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

rho       = 0.999
beta      = 0

# reference sinogram SPDHG recon
cost_spdhg_sino = np.zeros(niter)

proj.init_subsets(nsubsets)
cbs = {'cost':cost_spdhg_sino}
ref = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
            fwhm = fwhm, gamma = 1/img.max(), rho = rho, verbose = True, 
            beta = beta, callback = _cb, callback_kwargs = cbs)
proj.init_subsets(1)


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
img_shape = tuple(lmproj.img_dim)

# estimate the norm of the pet fwd operator

x = np.random.rand(*img_shape)

for i in range(10):
  x_fwd = ppp.pet_fwd_model(x, proj, attn_sino, sens_sino, 0, fwhm = fwhm)
  x     = ppp.pet_back_model(x_fwd, proj, attn_sino, sens_sino, 0, fwhm = fwhm)

  norm  = np.linalg.norm(x)
  print(np.sqrt(norm))

  x /= norm

pet_norm = np.sqrt(norm) / nsubsets

#-----------------------------------------------------------------------------------------------------

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

xs     = []
gammas = np.array([0.5/img.max(),1/img.max(),2/img.max()])

cost_spdhg_lm = np.zeros((len(gammas),niter))

for ig,gamma in enumerate(gammas):

  #S_i = []
  #ones_img = np.ones(img_shape, dtype = np.float32)
  #for i in range(nsubsets):
  #  ss = slice(i,None,nsubsets)
  #  tmp = (gamma*rho) / pet_fwd_model_lm(ones_img, lmproj, events[ss,:5], 
  #                                       attn_list[ss], sens_list[ss], fwhm = fwhm)
  #  tmp[tmp == np.inf] = tmp[tmp != np.inf].max()
  #  S_i.append(tmp)
  #
  #if p_g > 0:
  #  # calculate S for the gradient operator
  #  S_g = (gamma*rho/grad_norm)
  #
  #
  #if p_g == 0:
  #  T_i = np.zeros((1,) + img_shape, dtype = np.float32)
  #else:
  #  T_i = np.zeros((2,) + img_shape, dtype = np.float32)
  #  T_i[1,...] = rho*p_g/(gamma*grad_norm)
  #
  #
  #ones_sino = np.ones(proj.sino_params.shape, dtype = np.float32)
  #
  #tmp = pet_back_model(ones_sino, proj, attn_sino, sens_sino, 0, fwhm = fwhm)
  #T_i[0,...] = (rho*p_p/gamma) / tmp  
  #
  #T = T_i.min(axis = 0)

  S_i = (gamma*rho/pet_norm)*np.ones(nsubsets)
  T   = rho*p_p/(gamma*pet_norm)
  
  #--------------------------------------------------------------------------------------------
  # initialize variables
  x = np.zeros(img_shape, dtype = np.float32)
  y = np.zeros(events.shape[0], dtype = np.float32)
  
  z    = pet_back_model((em_sino == 0).astype(np.float32), proj, attn_sino, sens_sino, 0, fwhm = fwhm)
  zbar = z.copy()
  
  # allocate arrays for gradient operations
  x_grad      = np.zeros((T.ndim,) + img_shape, dtype = np.float32)
  y_grad      = np.zeros((T.ndim,) + img_shape, dtype = np.float32)
  y_grad_plus = np.zeros((T.ndim,) + img_shape, dtype = np.float32)
  
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
                                                  fwhm = fwhm) + contam_list[ss])/events[ss,5]
  
        # apply the prox for the dual of the poisson logL
        #y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]*events[ss,5]))
        y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]))
  
        dz = pet_back_model_lm((y_plus - y[ss])/events[ss,5], lmproj, events[ss,:5], 
                               attn_list[ss], sens_list[ss], fwhm = fwhm)
  
        # update variables
        z = z + dz
        y[ss] = y_plus.copy()
        zbar = z + dz/p_p
      else:
        print(f'iteration {it + 1} step {iss} gradient update')
  
        grad(x, x_grad)
        y_grad_plus = (y_grad + S_g*x_grad).reshape(T.ndim,-1)
  
        # proximity operator for dual of TV
        gnorm = np.linalg.norm(y_grad_plus, axis = 0)
        y_grad_plus /= np.maximum(np.ones(gnorm.shape, np.float32), gnorm / beta)
        y_grad_plus = y_grad_plus.reshape(x_grad.shape)
  
        dz = -1*div(y_grad_plus - y_grad)
  
        # update variables
        z = z + dz
        y_grad = y_grad_plus.copy()
        zbar = z + dz/p_g

    # calculate the cost
    cost_spdhg_lm[ig,it] = calc_cost(x)

  xs.append(x)

xs = np.array(xs)

#-----------------------------------------------------------------------------------------------------
vmax = 1.2*img.max()
sigs = 4.5/(2.35*voxsize)

fig, ax = plt.subplots(2,len(gammas) + 1, figsize = (4*(len(gammas) + 1),8))
ax[0,0].imshow(ref.squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
ax[1,0].imshow(gaussian_filter(ref, sigs).squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
ax[0,0].set_title('sino SPDHG', fontsize = 'medium')
ax[1,0].set_title('p.s. sino SPDHG', fontsize = 'medium')

for i,gam in enumerate(gammas):
  ax[0,i+1].imshow(xs[i,...].squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,i+1].imshow(gaussian_filter(xs[i,...], sigs).squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[0,i+1].set_title(f'LM SPDHG {gam:.1e}', fontsize = 'medium')
  ax[1,i+1].set_title(f'p.s. LM SPDHG {gam:.1e}', fontsize = 'medium')

for axx in ax.ravel():
  axx.set_axis_off()

fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots()
ax2.plot(cost_spdhg_sino[5:], label = 'SPDHG SINO')
for ig, gam in enumerate(gammas):
  ax2.plot(cost_spdhg_lm[ig,5:], label = f'SPDHG LM {gam:.2e}')
ax2.legend()
fig2.tight_layout()
fig2.show()
