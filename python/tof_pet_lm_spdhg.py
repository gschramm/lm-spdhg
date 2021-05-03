# small demo for sinogram TOF OS-MLEM
#TODO: - prox for dual of L1 and L2 form

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom
from pyparallelproj.utils import grad, div
from pyparallelproj.algorithms import spdhg

from algorithms import spdhg_lm

from scipy.ndimage import gaussian_filter
import numpy as np
import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 1,   type = int)
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 20,  type = int)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
parser.add_argument('--beta',  help = 'TV weight',  default = 0, type = float)
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
beta          = args.beta

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

#-----------------------------------------------------------------------------------------------------
def calc_cost(x):
  cost = 0

  ns = proj.nsubsets
  proj.init_subsets(1)

  for i in range(proj.nsubsets):
    # get the slice for the current subset
    ss = proj.subset_slices[i]
    exp = ppp.pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm) + contam_sino[ss]
    cost += (exp - em_sino[ss]*np.log(exp)).sum()

  proj.init_subsets(ns)

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

  if 'psnr' in kwargs:
    MSE = ((x - kwargs['xref'])**2).mean()
    kwargs['psnr'][it-1] = 20*np.log10(kwargs['xref'].max()/np.sqrt(MSE))


#-------------------------------------------------------------------------------------
# rerfence reconstruction using PDHG

niter_ref = 3000

ref_fname = os.path.join('data', f'PDHG_{phantom}_TVbeta_{beta:.1E}_niter_{niter_ref}_counts_{counts:.1E}_seed_{seed}.npz')
if os.path.exists(ref_fname):
  tmp = np.load(ref_fname)
  ref_recon = tmp['ref_recon']
  cost_ref  = tmp['cost_ref']
else:
  cost_ref  = np.zeros(niter_ref)
  
  ns = proj.nsubsets
  proj.init_subsets(1)
  ref_recon = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter_ref,
                    gamma = 1/img.max(), fwhm = fwhm, verbose = True, 
                    beta = beta, callback = _cb, callback_kwargs = {'cost': cost_ref})
  proj.init_subsets(ns)

  np.savez(ref_fname, ref_recon = ref_recon, cost_ref = cost_ref)


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


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

base_str = f'{phantom}_counts_{counts:.1E}_beta_{beta:.1E}_niter_{niter_ref}_{niter}_nsub_{nsubsets}'

rho    = 0.999
gammas = np.array([0.1,0.3,1,3,10,30]) / img.max()

cost_spdhg_sino = np.zeros((len(gammas),niter))
cost_spdhg_lm   = np.zeros((len(gammas),niter))

psnr_spdhg_sino = np.zeros((len(gammas),niter))
psnr_spdhg_lm   = np.zeros((len(gammas),niter))

x_sino = np.zeros((len(gammas),) + img.shape)
x_lm   = np.zeros((len(gammas),) + img.shape)

for ig,gamma in enumerate(gammas):
  # sinogram SPDHG
  proj.init_subsets(nsubsets)
  cbs = {'cost':cost_spdhg_sino[ig,:],'psnr':psnr_spdhg_sino[ig,:],'xref':ref_recon}
  x_sino[ig,...] = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
                         fwhm = fwhm, gamma = gamma, rho = rho, verbose = True, 
                         beta = beta, callback = _cb, callback_kwargs = cbs)
  proj.init_subsets(1)


  # LM SPDHG
  cblm = {'cost':cost_spdhg_lm[ig,:],'psnr':psnr_spdhg_lm[ig,:],'xref':ref_recon}
  x_lm[ig,...] = spdhg_lm(events, multi_index,
                          em_sino, attn_sino, sens_sino, contam_sino, 
                          proj, lmproj, niter, nsubsets,
                          fwhm = fwhm, gamma = gamma, rho = rho, verbose = True, 
                          beta = beta, callback = _cb, callback_kwargs = cblm)
#-----------------------------------------------------------------------------------------------------
# save the results
ofile = os.path.join('data',f'{base_str}.npz')
np.savez(ofile, ref_recon = ref_recon, cost_ref = cost_ref,
                cost_spdhg_sino = cost_spdhg_sino, psnr_spdhg_sino = psnr_spdhg_sino, 
                cost_spdhg_lm   = cost_spdhg_lm, psnr_spdhg_lm = psnr_spdhg_lm, 
                x_sino = x_sino, x_lm = x_lm, gammas = gammas)


# show the results

vmax = 1.2*img.max()

fig, ax = plt.subplots(2,len(gammas)+1, figsize = (3*(len(gammas)+1),6))
for i,gam in enumerate(gammas):
  ax[0,i].imshow(x_sino[i,...].squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,i].imshow(x_lm[i,...].squeeze(),   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)

  ax[0,i].set_title(f'SPDHG {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  ax[1,i].set_title(f'LM SPDHG {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))

ax[0,-1].imshow(ref_recon.squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
ax[1,-1].imshow(img.squeeze(),   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
ax[0,-1].set_title(f'reference PDHG', fontsize = 'medium')
ax[1,-1].set_title(f'ground truth', fontsize = 'medium')

for axx in ax.ravel():
  axx.set_axis_off()

fig.tight_layout()
fig.savefig(os.path.join('data',f'{base_str}.png'))
fig.show()



c_ref = cost_ref.min()
c_0   = calc_cost(np.zeros(img.shape, dtype = np.float32))
n     = c_0 - c_ref
ni    = np.arange(niter) + 1

fig2, ax2 = plt.subplots(1,2, figsize = (10,5))
for ig, gam in enumerate(gammas):
  col = plt.get_cmap("tab10")(ig)
  ax2[0].semilogy(ni, (cost_spdhg_sino[ig,:] - c_ref)/n, '-.', ms = 5,
               label = f'SINO {gam:.2e}',color = col)
  ax2[0].semilogy(ni, (cost_spdhg_lm[ig,:] - c_ref)/n, '-v', ms = 5, 
               label = f'LM {gam:.2e}', color = col)
  ax2[1].plot(ni, psnr_spdhg_sino[ig,:], '-.', ms = 5, color = col)
  ax2[1].plot(ni, psnr_spdhg_lm[ig,:], '-v', ms = 5, color = col)

ax2[0].grid(ls = ':')
ax2[0].set_xlabel('iteration')
ax2[0].set_ylabel('relative cost')
ax2[1].grid(ls = ':')
ax2[1].set_xlabel('iteration')
ax2[1].set_ylabel('PSNR to reference')

handles, labels = ax2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', ncol = len(gammas), fontsize = 'small')

fig2.tight_layout()
fig2.savefig(os.path.join('data',f'{base_str}_metrics.pdf'))
fig2.savefig(os.path.join('data',f'{base_str}_metrics.png'))
fig2.show()



