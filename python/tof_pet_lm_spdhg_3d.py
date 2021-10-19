#TODO: - prox for dual of L1 and L2 form

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom
from pyparallelproj.utils import GradientOperator
from pyparallelproj.algorithms import spdhg, osem, osem_lm

from algorithms import spdhg_lm
from utils import  plot_lm_spdhg_results

from scipy.ndimage import gaussian_filter
import numpy as np
import argparse

from time import time

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',   help = 'counts to simulate',    default = 3e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 10,  type = int)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 56,  type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
parser.add_argument('--beta',  help = 'TV weight',  default = 3e-1, type = float)
parser.add_argument('--prior',  help = 'prior',  default = 'TV', choices = ['TV','DTV'])
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts        = args.counts
niter         = args.niter
nsubsets      = args.nsubsets
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
seed          = args.seed
beta          = args.beta
prior         = args.prior

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,9]),
                                       nmodules             = np.array([28,3]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n0      = 120
n1      = 120
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup the ground truth image

img = np.zeros((n0,n1,n2), dtype = np.float32)
img[(n0//8):(-n0//8),(n1//8):(-n1//8),:] = 1

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# setup an attenuation image, projector takes voxel size into account!
# so mu should be in 1/mm
att_img = (img > 0) * 0.01

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
rimg  = np.random.rand(*img.shape).astype(np.float32)

for i in range(5):
  rimg_fwd = ppp.pet_fwd_model(rimg, proj, attn_sino, sens_sino, 0, fwhm = fwhm)
  rimg     = ppp.pet_back_model(rimg_fwd, proj, attn_sino, sens_sino, 0, fwhm = fwhm)

  pnsq     = np.linalg.norm(rimg)
  rimg    /= pnsq
  print(np.sqrt(pnsq))


# scale the sensitivity and the image such that the norm of the PET fwd operator is approx 
# eual to the norm of the 3D gradient operator
sens_sino /= (np.sqrt(pnsq)/np.sqrt(12))

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

print('\nCreating LM events')

events, multi_index = sino_params.sinogram_to_listmode(em_sino, 
                                                       return_multi_index = True,
                                                       return_counts = False)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

gamma = 1. / img.max()

print('\nSinogram SPDHG')

proj.init_subsets(nsubsets)
t0 = time()
x_sino = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, 3,
               fwhm = fwhm, gamma = gamma, verbose = True, 
               beta = beta, grad_operator = grad_operator)
t1 = time()

# LM SPDHG
print('\nLM SPDHG')

proj.init_subsets(1)

t2 = time()
x_lm = spdhg_lm(events, multi_index, attn_sino, sens_sino, contam_sino, 
                proj, lmproj, 3, nsubsets, fwhm = fwhm, gamma = gamma, verbose = True, 
                beta = beta, grad_operator = grad_operator)
t3 = time()

print(t1-t0,t3-t2)

import pymirc.viewer as pv
vi = pv.ThreeAxisViewer([img,x_sino,x_lm], imshow_kwargs = {'vmin':0, 'vmax':1.2*img.max()})
