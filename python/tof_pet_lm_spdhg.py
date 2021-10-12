#TODO: - prox for dual of L1 and L2 form

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom
from pyparallelproj.utils import GradientOperator
from pyparallelproj.algorithms import spdhg

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
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 224,  type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
parser.add_argument('--beta',  help = 'TV weight',  default = 3e-3, type = float)
parser.add_argument('--prior',  help = 'prior',  default = 'TV', choices = ['TV','DTV'])
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
                                    voxsize = voxsize, img_origin = img_origin)
sino_shape_nt  = sino_params_nt.shape

attn_sino = np.zeros(sino_shape_nt, dtype = np.float32)
attn_sino = np.exp(-proj_nt.fwd_project(att_img))

# generate the sensitivity sinogram
sens_sino = np.ones(sino_shape_nt, dtype = np.float32)

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
    x_grad = grad_operator.fwd(x)
    cost += beta*np.linalg.norm(x_grad, axis = 0).sum()

  return cost

def _cb(x, **kwargs):
  it = kwargs.get('iteration',0)
  it_early1 = kwargs.get('it_early1',-1)
  it_early2 = kwargs.get('it_early2',-1)
  it_early3 = kwargs.get('it_early3',-1)
  it_early4 = kwargs.get('it_early4',-1)

  if it_early1 == it:
    if 'x_early1' in kwargs:
      kwargs['x_early1'][:] = x

  if it_early2 == it:
    if 'x_early2' in kwargs:
      kwargs['x_early2'][:] = x

  if it_early3 == it:
    if 'x_early3' in kwargs:
      kwargs['x_early3'][:] = x

  if it_early4 == it:
    if 'x_early4' in kwargs:
      kwargs['x_early4'][:] = x


  if 'cost' in kwargs:
    kwargs['cost'][it-1] = calc_cost(x)

  if 'psnr' in kwargs:
    MSE = ((x - kwargs['xref'])**2).mean()
    kwargs['psnr'][it-1] = 20*np.log10(kwargs['xref'].max()/np.sqrt(MSE))


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
                    gamma = 1/img.max(), fwhm = fwhm, verbose = True, 
                    beta = beta, callback = _cb, callback_kwargs = {'cost': cost_ref},
                    grad_operator = grad_operator)


  proj.init_subsets(ns)
  np.savez(ref_fname, ref_recon = ref_recon, cost_ref = cost_ref)

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

gammas = np.array([1.]) / img.max()

cost_spdhg_sino = np.zeros((len(gammas),niter))
cost_spdhg_lm   = np.zeros((len(gammas),niter))

psnr_spdhg_sino = np.zeros((len(gammas),niter))
psnr_spdhg_lm   = np.zeros((len(gammas),niter))

x_sino = np.zeros((len(gammas),) + img.shape)
x_lm   = np.zeros((len(gammas),) + img.shape)

x_early1_sino = np.zeros((len(gammas),) + img.shape)
x_early2_sino = np.zeros((len(gammas),) + img.shape)
x_early3_sino = np.zeros((len(gammas),) + img.shape)
x_early4_sino = np.zeros((len(gammas),) + img.shape)
x_early1_lm   = np.zeros((len(gammas),) + img.shape)
x_early2_lm   = np.zeros((len(gammas),) + img.shape)
x_early3_lm   = np.zeros((len(gammas),) + img.shape)
x_early4_lm   = np.zeros((len(gammas),) + img.shape)

for ig,gamma in enumerate(gammas):
  # sinogram SPDHG
  proj.init_subsets(nsubsets)
  cbs = {'cost':cost_spdhg_sino[ig,:],'psnr':psnr_spdhg_sino[ig,:],'xref':ref_recon,
         'it_early1':1, 'x_early1':x_early1_sino[ig,:],
         'it_early2':2, 'x_early2':x_early2_sino[ig,:],
         'it_early3':5, 'x_early3':x_early3_sino[ig,:],
         'it_early4':10,'x_early4':x_early4_sino[ig,:]}

  x_sino[ig,...] = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
                         fwhm = fwhm, gamma = gamma, verbose = True, 
                         beta = beta, callback = _cb, callback_kwargs = cbs,
                         grad_operator = grad_operator)

  # LM SPDHG
  proj.init_subsets(1)
  cblm = {'cost':cost_spdhg_lm[ig,:],'psnr':psnr_spdhg_lm[ig,:],'xref':ref_recon,
         'it_early1':1, 'x_early1':x_early1_lm[ig,:],
         'it_early2':2, 'x_early2':x_early2_lm[ig,:],
         'it_early3':5, 'x_early3':x_early3_lm[ig,:],
         'it_early4':10,'x_early4':x_early4_lm[ig,:]}

  x_lm[ig,...] = spdhg_lm(events, multi_index, attn_sino, sens_sino, contam_sino, 
                          proj, lmproj, niter, nsubsets,
                          fwhm = fwhm, gamma = gamma, verbose = True, 
                          beta = beta, callback = _cb, callback_kwargs = cblm,
                          grad_operator = grad_operator)

#-----------------------------------------------------------------------------------------------------
# calculate cost of initial recon (image full or zeros)
c_0   = calc_cost(np.zeros(ref_recon.shape, dtype = np.float32))

# save the results
ofile = os.path.join('data',f'{base_str}_niter_{niter}_nsub_{nsubsets}.npz')
np.savez(ofile, ref_recon = ref_recon, cost_ref = cost_ref,
                cost_spdhg_sino = cost_spdhg_sino, psnr_spdhg_sino = psnr_spdhg_sino, 
                cost_spdhg_lm   = cost_spdhg_lm, psnr_spdhg_lm = psnr_spdhg_lm, 
                x_sino = x_sino, x_lm = x_lm,
                x_early1_sino = x_early1_sino, x_early1_lm = x_early1_lm,
                x_early2_sino = x_early2_sino, x_early2_lm = x_early2_lm,
                x_early3_sino = x_early3_sino, x_early3_lm = x_early3_lm,
                x_early4_sino = x_early4_sino, x_early4_lm = x_early4_lm,
                gammas = gammas, img = img, c_0 = c_0)

#-----------------------------------------------------------------------------------------------------
# plot the results
plot_lm_spdhg_results(ofile)
