# small demo for sinogram TOF OS-MLEM

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom
from pyparallelproj.models import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm
from pyparallelproj.utils import grad, div

import numpy as np
import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 4,   type = int)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 28,  type = int)
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

proj.init_subsets(nsubsets)

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

events      = []
contam_list = []
sens_list   = []
attn_list   = []
fwd_ones    = []

# calculate the "step sizes" S_i, T_i  for the projector
ones_img = np.ones(tuple(lmproj.img_dim), dtype = np.float32)

for i in range(nsubsets):
  ss = proj.subset_slices[i]
  tmp, multi_index = sino_params.sinogram_to_listmode(em_sino[ss], 
                                                      subset = i, nsubsets = nsubsets,
                                                      return_multi_index = True,
                                                      return_counts = True)

  events.append(tmp)
  
  contam_list.append(contam_sino[ss][multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]])
  sens_list.append(sens_sino[ss][multi_index[:,0],multi_index[:,1],multi_index[:,2],0])
  attn_list.append(attn_sino[ss][multi_index[:,0],multi_index[:,1],multi_index[:,2],0])

  # fwd project a ones image for the S_i step sizes
  fwd_ones.append(pet_fwd_model(ones_img, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)[multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]])

# backproject the subset events in LM and sino mode as a cross check
#back_imgs  = np.zeros((nsubsets,) + tuple(lmproj.img_dim))
#back_imgs2 = np.zeros((nsubsets,) + tuple(lmproj.img_dim))
#
#for i in range(nsubsets):
#  subset_events = events[i][:,:5]
#  back_imgs[i,...] = lmproj.back_project(np.ones(subset_events.shape[0], dtype = np.float32), subset_events)
#  back_imgs2[i,...] = proj.back_project(em_sino[proj.subset_slices[i]], subset = i)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
gamma     = 1/img.max()
rho       = 0.999
beta      = 0
nsubsets  = proj.nsubsets

img_shape = tuple(lmproj.img_dim)

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

S_i = []
for i in range(nsubsets):
  tmp = (gamma*rho) / fwd_ones[i]
  tmp[tmp == np.inf] = tmp[tmp != np.inf].max()
  S_i.append(tmp)

if p_g > 0:
  # calculate S for the gradient operator
  S_g = (gamma*rho/grad_norm)


if p_g == 0:
  T_i = np.zeros((nsubsets,) + img_shape, dtype = np.float32)
else:
  T_i = np.zeros(((nsubsets+1),) + img_shape, dtype = np.float32)
  T_i[-1,...] = rho*p_g/(gamma*grad_norm)

for i in range(nsubsets):
  # get the slice for the current subset
  ss = proj.subset_slices[i]
  # generate a subset sinogram full of ones
  ones_sino = np.ones(proj.subset_sino_shapes[i] , dtype = np.float32)

  tmp = pet_back_model(ones_sino, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
  T_i[i,...] = (rho*p_p/gamma) / tmp  
                                                       
# take the element-wise min of the T_i's of all subsets
T = T_i.min(axis = 0)
