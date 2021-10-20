#TODO: - prox for dual of L1 and L2 form

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom
from pyparallelproj.utils import GradientOperator, GradientNorm
from pyparallelproj.algorithms import spdhg, osem_lm_emtv

from algorithms import spdhg_lm
from utils import  plot_lm_spdhg_results

from scipy.ndimage import gaussian_filter
import numpy as np
import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6,  type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 100,  type = int)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 28,   type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5,  type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
parser.add_argument('--beta',  help = 'TV weight',  default = 3e-2, type = float)
parser.add_argument('--gamma',  help = 'relative gamma',  default = 1., type = float)
parser.add_argument('--prior',  help = 'prior',  default = 'TV', choices = ['TV','DTV'])
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts        = args.counts
niter         = args.niter
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
phantom       = args.phantom
seed          = args.seed
beta          = args.beta
prior         = args.prior
gamma         = args.gamma
nsubsets      = args.nsubsets

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
if phantom == 'brain2d':
  n   = 128
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = brain2d_phantom(n = n)
  for i2 in range(n2):
    img[:,:,i2] = tmp

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# setup an attenuation image, projector takes voxel size into account!
# so mu should be in 1/mm
att_img = (img > 0) * 0.01

# generate nonTOF sinogram parameters and the nonTOF projector for attenuation projection
sino_params_nt = ppp.PETSinogramParameters(scanner)
proj_nt        = ppp.SinogramProjector(scanner, sino_params_nt, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin)

attn_sino = np.exp(-proj_nt.fwd_project(att_img))

# generate the sensitivity sinogram
sens_sino = np.ones(sino_params_nt.shape, dtype = np.float32)

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# power iterations to estimte norm of PET fwd operator
rimg = np.random.rand(*img.shape).astype(np.float32)
for i in range(40):
  rimg_fwd = ppp.pet_fwd_model(rimg, proj, attn_sino, sens_sino, 0, fwhm = fwhm)
  rimg     = ppp.pet_back_model(rimg_fwd, proj, attn_sino, sens_sino, 0, fwhm = fwhm)

  pnsq     = np.linalg.norm(rimg)
  rimg    /= pnsq
  print(np.sqrt(pnsq))

# scale the sensitivity and the image such that the norm of the PET fwd operator is approx 
# eual to the norm of the 2D gradient operator
sens_sino /= (np.sqrt(pnsq)/np.sqrt(8))

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


G0 = GradientOperator()
if prior == 'DTV':
  grad_operator  = GradientOperator(joint_grad_field = G0.fwd(-(img**0.5)))
else:
  grad_operator  = GradientOperator()

grad_norm = GradientNorm(name = 'l2_l1', beta = beta)

#-----------------------------------------------------------------------------------------------------
def calc_cost(x):
  cost = 0

  for i in range(proj.nsubsets):
    # get the slice for the current subset
    ss = proj.subset_slices[i]
    exp = ppp.pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm) + contam_sino[ss]
    cost += (exp - em_sino[ss]*np.log(exp)).sum()

  if grad_norm.beta > 0:
    cost += grad_norm.eval(grad_operator.fwd(x))

  return cost

def _cb(x, **kwargs):
  it = kwargs.get('iteration',0)

  if 'intermed' in kwargs:
    kwargs['intermed'][it-1,...] = x

  if 'cost' in kwargs:
    kwargs['cost'][it-1] = calc_cost(x)

  if 'psnr' in kwargs:
    MSE = ((x - kwargs['xref'])**2).mean()
    kwargs['psnr'][it-1] = 20*np.log10(kwargs['xref'].max()/np.sqrt(MSE))


#-------------------------------------------------------------------------------------

# create a listmode projector for the LM MLEM iterations
lmproj = ppp.LMProjector(proj.scanner, proj.img_dim, voxsize = proj.voxsize, 
                         img_origin = proj.img_origin,
                         tof = proj.tof, sigma_tof = proj.sigma_tof, tofbin_width = proj.tofbin_width,
                         n_sigmas = proj.nsigmas)

# generate list mode events and the corresponting values in the contamination and sensitivity
# for every subset sinogram

# events is a list of nsubsets containing (nevents,6) arrays where
# [:,0:2] are the transaxial/axial crystal index of the 1st detector
# [:,2:4] are the transaxial/axial crystal index of the 2nd detector
# [:,4]   are the event TOF bins

events, multi_index = sino_params.sinogram_to_listmode(em_sino, 
                                                       return_multi_index = True,
                                                       return_counts = False)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

attn_list   = attn_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
sens_list   = sens_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
contam_list = contam_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]]

# LM-OSEM EM-TV
proj.init_subsets(1)
ones_sino = np.ones(proj.sino_params.shape , dtype = np.float32)
sens_img  = ppp.pet_back_model(ones_sino, proj, attn_sino, sens_sino, 0, fwhm = fwhm)
 
xinit = osem_lm_emtv(events, attn_list, sens_list, contam_list, lmproj, sens_img, 2, 28, 
                     grad_operator = grad_operator, grad_norm = grad_norm,
                     fwhm = fwhm, verbose = True)

proj.init_subsets(1)
x_fwd = ppp.pet_fwd_model(xinit, proj, attn_sino, sens_sino, 0, fwhm = fwhm) + contam_sino
yinit = 1 - (em_sino / x_fwd)

#y_init_grad = grad_norm.beta*np.tanh(grad_operator.fwd(xinit/xinit.max()))

#-----------------------------------------------------------------------------------------------------

cost_emtv = np.zeros(niter)
x_emtv_intermed = np.zeros((niter,) + img.shape) 

x_emtv = osem_lm_emtv(events, attn_list, sens_list, contam_list, lmproj, sens_img, niter, nsubsets, 
                      fwhm = fwhm, verbose = True, callback = _cb,
                      grad_operator = grad_operator, grad_norm = grad_norm,
                      callback_kwargs = {'cost':cost_emtv, 'intermed': x_emtv_intermed})

#-----------------------------------------------------------------------------------------------------

cost_spdhg      = np.zeros(niter)
cost_spdhg_warm = np.zeros(niter)

x_spdhg_intermed = np.zeros((niter,) + img.shape) 
x_spdhg_warm_intermed = np.zeros((niter,) + img.shape) 

proj.init_subsets(nsubsets)
x_sino = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
               fwhm = fwhm, gamma = gamma/img.max(), verbose = True, ystart = None, xstart = None,
               grad_operator = grad_operator, grad_norm = grad_norm,
               callback = _cb, callback_kwargs = {'cost':cost_spdhg, 'intermed': x_spdhg_intermed})

x_sino_warm = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
                    fwhm = fwhm, gamma = gamma/img.max(), verbose = True, ystart = yinit, xstart = xinit,
                    grad_operator = grad_operator, grad_norm = grad_norm, 
                    callback = _cb, callback_kwargs = {'cost':cost_spdhg_warm, 'intermed': x_spdhg_warm_intermed})

#-----------------------------------------------------------------------------------------------------

ist = min(niter-10,1000)

min_cost = min(cost_spdhg_warm.min(),cost_spdhg.min(), cost_emtv.min())
max_cost = max(cost_spdhg_warm[ist:].max(),cost_spdhg[ist:].max(), cost_emtv[ist:].max())

fig, ax = plt.subplots(2,4, figsize = (16,8))
ax[0,0].imshow(x_sino.squeeze(), vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[0,0].set_title('SPDHG cold')
ax[0,1].imshow(x_sino_warm.squeeze(), vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[0,1].set_title('SPDHG warm')
ax[0,2].imshow(x_sino.squeeze() - x_sino_warm.squeeze(), vmin = -0.05*img.max(), 
               vmax = 0.05*img.max(), cmap = plt.cm.bwr)
ax[0,2].set_title('SPDHG cold - SPDHG warm')

ax[0,3].plot(cost_spdhg, label = 'SPDHG')
ax[0,3].plot(cost_spdhg_warm, label = 'SPDHG warm')
ax[0,3].plot(cost_emtv, label = 'EM-TV')
ax[0,3].set_ylim(min_cost - 0.05*(max_cost - min_cost), max_cost + 0.05*(max_cost - min_cost))
ax[0,3].grid(ls = ':')
ax[0,3].legend()
ax[0,3].set_xlabel('iteration')
ax[0,3].set_ylabel('cost')

ax[1,0].imshow(xinit.squeeze(), vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[1,0].set_title('warm init.')
ax[1,1].imshow(x_emtv.squeeze(), vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[1,1].set_title('EM-TV')
ax[1,2].imshow(x_emtv.squeeze() - x_sino_warm.squeeze(), vmin = -0.05*img.max(), 
               vmax = 0.05*img.max(), cmap = plt.cm.bwr)
ax[1,2].set_title('EM-TV - SPDHG warm')

ax[1,3].set_axis_off()

fig.tight_layout()
fig.savefig('test.png')
fig.show()


import pymirc.viewer as pv
vi = pv.ThreeAxisViewer([np.swapaxes(x_spdhg_intermed.squeeze(),0,2),
                         np.swapaxes(x_spdhg_warm_intermed.squeeze(),0,2),
                         np.swapaxes(x_emtv_intermed.squeeze(),0,2)], 
                         imshow_kwargs = {'vmax':1.2*img.max()})
