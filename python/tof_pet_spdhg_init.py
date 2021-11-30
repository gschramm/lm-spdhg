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
parser.add_argument('--counts',       help = 'counts to simulate',        default = 1e6, type = float)
parser.add_argument('--niter',        help = 'number of iterations',      default = 20,  type = int)
parser.add_argument('--nsubsets',     help = 'number of subsets',         default = 56,  type = int)
parser.add_argument('--fwhm_mm',      help = 'psf modeling FWHM mm',      default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm', help = 'psf for data FWHM mm',      default = 4.5, type = float)
parser.add_argument('--phantom',      help = 'phantom to use',            default = 'brain2d')
parser.add_argument('--seed',         help = 'seed for random generator', default = 1, type = int)
parser.add_argument('--beta',         help = 'TV weight',                 default = 3e-2, type = float)
parser.add_argument('--prior',        help = 'prior',                     default = 'TV', choices = ['TV','DTV'])
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts        = args.counts
niter         = args.niter
nsubsets      = args.nsubsets
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
phantom       = args.phantom
seed          = args.seed
beta          = args.beta
prior         = args.prior

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

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# create the attenuation sinogram
proj.set_tof(False)
attn_sino = np.exp(-proj.fwd_project(att_img))
proj.set_tof(True)
# generate the sensitivity sinogram
sens_sino = np.ones(proj.sino_params.nontof_shape, dtype = np.float32)

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

# forward project the image
img_fwd = ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, fwhm = fwhm_data)

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
  exp  = ppp.pet_fwd_model(x, proj, attn_sino, sens_sino, fwhm = fwhm) + contam_sino
  cost = (exp - em_sino*np.log(exp)).sum()

  if grad_norm.beta > 0:
    cost += grad_norm.eval(grad_operator.fwd(x))

  return cost

def _cb(x, **kwargs):
  it = kwargs.get('iteration',0)

  if it == 3:
    if 'x_early' in kwargs:
      kwargs['x_early'][:] = x

  if 'cost' in kwargs:
    kwargs['cost'][it-1] = calc_cost(x)

  if 'psnr' in kwargs:
    MSE = ((x - kwargs['xref'])**2).mean()
    kwargs['psnr'][it-1] = 20*np.log10(kwargs['xref'].max()/np.sqrt(MSE))


#-------------------------------------------------------------------------------------

# generate list mode events and the corresponting values in the contamination and sensitivity
# for every subset sinogram

# events is a list of nsubsets containing (nevents,6) arrays where
# [:,0:2] are the transaxial/axial crystal index of the 1st detector
# [:,2:4] are the transaxial/axial crystal index of the 2nd detector
# [:,4]   are the event TOF bins

events, multi_index = sino_params.sinogram_to_listmode(em_sino, 
                                                       return_multi_index = True,
                                                       return_counts = False)

attn_list   = attn_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
sens_list   = sens_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
contam_list = contam_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]]

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

sens_img  = ppp.pet_back_model(np.ones(proj.sino_params.shape, dtype = np.float32), 
                               proj, attn_sino, sens_sino, fwhm = fwhm)

xinit = osem_lm_emtv(events, attn_list, sens_list, contam_list, proj, sens_img, 1, 28, 
                     grad_operator = grad_operator, grad_norm = grad_norm,
                     fwhm = fwhm, verbose = True)

yinit = 1 - (em_sino / (ppp.pet_fwd_model(xinit, proj, attn_sino, sens_sino, fwhm = fwhm) + contam_sino))

#-------------------------------------------------------------------------------------
# rerfence reconstruction using PDHG

niter_ref = 20000

base_str = f'{phantom}_counts_{counts:.1E}_seed_{seed}_beta_{beta:.1E}_prior_{prior}_niter_ref_{niter_ref}_fwhm_{fwhm_mm:.1f}_{fwhm_data_mm:.1f}'


ref_fname = os.path.join('data', f'{base_str}_ref.npz')

if os.path.exists(ref_fname):
  tmp = np.load(ref_fname, allow_pickle = True)
  ref_recon = tmp['ref_recon']
  cost_ref  = tmp['cost_ref']
else:
  ns = proj.nsubsets
  proj.init_subsets(1)

  # do long PDHG recon
  cost_ref  = np.zeros(niter_ref)
  ref_recon = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter_ref,
                    gamma = 3./img.max(), fwhm = fwhm, verbose = True, 
                    xstart = xinit, ystart = yinit,
                    callback = _cb, callback_kwargs = {'cost': cost_ref},
                    grad_operator = grad_operator, grad_norm = grad_norm)


  proj.init_subsets(ns)
  np.savez(ref_fname, ref_recon = ref_recon, cost_ref = cost_ref)


#-----------------------------------------------------------------------------------------------------

y_init_grad = grad_norm.beta*np.sign(grad_operator.fwd(xinit))

gamma = 3. / gaussian_filter(xinit.squeeze(),2).max()

cost_spdhg_sino = np.zeros((2,niter))
psnr_spdhg_sino = np.zeros((2,niter))

# sinogram SPDHG
proj.init_subsets(nsubsets)
cbs1 = {'cost':cost_spdhg_sino[0,:],'psnr':psnr_spdhg_sino[0,:],'xref':ref_recon, 'x_early':np.zeros(img.shape)}
cbs2 = {'cost':cost_spdhg_sino[1,:],'psnr':psnr_spdhg_sino[1,:],'xref':ref_recon, 'x_early':np.zeros(img.shape)}

x1 = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
           fwhm = fwhm, gamma = gamma, verbose = True, 
           callback = _cb, callback_kwargs = cbs1,
           grad_operator = grad_operator, grad_norm = grad_norm)

x2 = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
           fwhm = fwhm, gamma = gamma, verbose = True, 
           xstart = xinit, ystart = yinit,
           callback = _cb, callback_kwargs = cbs2,
           grad_operator = grad_operator, grad_norm = grad_norm)

#----------------------------------------------------------------------------------------

c0    = calc_cost(xinit)
c_ref = cost_ref.min()

n = np.arange(1, niter+1)

plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(2,4, figsize = (7,5))

ax[0,0].semilogy(n, (cost_spdhg_sino[0,...] - c_ref)/(c0 - c_ref), label = 'cold start')
ax[0,0].semilogy(n, (cost_spdhg_sino[1,...] - c_ref)/(c0 - c_ref), label = 'warm start')
ax[0,0].set_xlabel('iteration')
ax[0,0].set_ylabel('relative cost')
ax[0,0].grid(ls = ':')
      
ax[1,0].plot(n, psnr_spdhg_sino[0,...], label = 'cold start')
ax[1,0].plot(n, psnr_spdhg_sino[1,...], label = 'warm start')
ax[1,0].set_xlabel('iteration')
ax[1,0].set_ylabel('PSNR')
ax[1,0].legend()
ax[1,0].grid(ls = ':')


ax[0,1].imshow(ref_recon.squeeze()[:,19:-19], vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[0,1].set_title('reference', fontsize = 'medium')
ax[1,1].imshow(xinit.squeeze()[:,19:-19], vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[1,1].set_title('initializer warm start', fontsize = 'medium')

ax[0,2].imshow(cbs1['x_early'].squeeze()[:,19:-19], vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[0,2].set_title('SPDHG cold start 3it', fontsize = 'medium')
ax[1,2].imshow(cbs2['x_early'].squeeze()[:,19:-19], vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[1,2].set_title('SPDHG warm start 3it', fontsize = 'medium')


ax[0,3].imshow(x1.squeeze()[:,19:-19], vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[0,3].set_title(f'SPDHG cold start {niter}it', fontsize = 'medium')
ax[1,3].imshow(x2.squeeze()[:,19:-19], vmax = 1.2*img.max(), cmap = plt.cm.Greys)
ax[1,3].set_title(f'SPDHG warm start {niter}it', fontsize = 'medium')


for axx in ax[:,1:].ravel():
  axx.set_axis_off()

fig.tight_layout()
#fig.savefig('SPDHG_cold_vs_warm_start.png')
fig.show()
