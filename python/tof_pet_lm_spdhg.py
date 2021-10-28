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
parser.add_argument('--niter',        help = 'number of iterations',      default = 100, type = int)
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
  it_early1 = kwargs.get('it_early1',-1)
  it_early2 = kwargs.get('it_early2',-1)
  it_early3 = kwargs.get('it_early3',-1)
  it_early4 = kwargs.get('it_early4',-1)

  if it == 10:
    if 'x_early' in kwargs:
      kwargs['x_early'][0,...] = x

  if it == 20:
    if 'x_early' in kwargs:
      kwargs['x_early'][1,...] = x

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

niter_ref = 10000

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

nsubsets = [56,112,224]
gamma    = 3. / gaussian_filter(xinit.squeeze(),2).max()

cost_spdhg_sino = np.zeros((len(nsubsets),niter))
cost_spdhg_lm   = np.zeros((len(nsubsets),niter))

psnr_spdhg_sino = np.zeros((len(nsubsets),niter))
psnr_spdhg_lm   = np.zeros((len(nsubsets),niter))

x_sino = np.zeros((len(nsubsets),) + img.shape)
x_lm   = np.zeros((len(nsubsets),) + img.shape)

x_sino_early = np.zeros((len(nsubsets),2) + img.shape)
x_lm_early   = np.zeros((len(nsubsets),2) + img.shape)

for iss, nss in enumerate(nsubsets):
  # sinogram SPDHG
  proj.init_subsets(nss)
  cbs = {'cost':cost_spdhg_sino[iss,:],'psnr':psnr_spdhg_sino[iss,:],'xref':ref_recon, 
         'x_early':x_sino_early[iss,...]}
  x_sino[iss,...] = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
                          fwhm = fwhm, gamma = gamma, verbose = True, 
                          xstart = xinit, ystart = yinit,
                          callback = _cb, callback_kwargs = cbs,
                          grad_operator = grad_operator, grad_norm = grad_norm)

  # LM SPDHG
  cblm = {'cost':cost_spdhg_lm[iss,:],'psnr':psnr_spdhg_lm[iss,:],'xref':ref_recon,
          'x_early':x_lm_early[iss,...]}
  x_lm[iss,...] = spdhg_lm(events, attn_list, sens_list, contam_list, sens_img,
                           proj, niter, nss, x0 = xinit,
                           fwhm = fwhm, gamma = gamma, verbose = True, 
                           callback = _cb, callback_kwargs = cblm,
                           grad_operator = grad_operator, grad_norm = grad_norm)

#-----------------------------------------------------------------------------------------------------
# run EM-TV with 1, 7 and nsubsets subsets for comparison

nsubsets_emtv = [1,7,28]

cost_emtv = np.zeros((len(nsubsets_emtv),niter))
psnr_emtv = np.zeros((len(nsubsets_emtv),niter))

x_emtv = np.zeros((len(nsubsets_emtv),) + img.shape)
x_emtv_early = np.zeros((len(nsubsets_emtv),2) + img.shape)

for ie, nsemtv in enumerate(nsubsets_emtv):
  cbemtv = {'cost':cost_emtv[ie,:],'psnr':psnr_emtv[ie,:],'xref':ref_recon, 'x_early':x_emtv_early[ie,...]}
  x_emtv[ie,...] = osem_lm_emtv(events, attn_list, sens_list, contam_list, proj, sens_img, niter, nsemtv, 
                                grad_operator = grad_operator, grad_norm = grad_norm,
                                callback = _cb, callback_kwargs = cbemtv, xstart = xinit,
                                fwhm = fwhm, verbose = True)

#-----------------------------------------------------------------------------------------------------
# calculate cost of initial recon (image full or zeros)
c_0 = calc_cost(xinit)

# save the results
ofile = os.path.join('data',f'{base_str}_niter_{niter}.npz')
np.savez(ofile, ref_recon = ref_recon, cost_ref = cost_ref,
                cost_spdhg_sino = cost_spdhg_sino, psnr_spdhg_sino = psnr_spdhg_sino, 
                cost_spdhg_lm = cost_spdhg_lm, psnr_spdhg_lm = psnr_spdhg_lm, 
                cost_emtv = cost_emtv, psnr_emtv = psnr_emtv, 
                x_sino = x_sino, x_lm = x_lm, x_emtv = x_emtv,
                x_sino_early = x_sino_early, x_lm_early = x_lm_early, x_emtv_early = x_emtv_early,
                nsubsets = nsubsets, nsubsets_emtv = nsubsets_emtv,
                gamma = gamma, img = img, c_0 = c_0)

#-----------------------------------------------------------------------------------------------------
# plot the results
plot_lm_spdhg_results(ofile)
