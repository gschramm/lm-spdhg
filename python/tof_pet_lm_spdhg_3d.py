#TODO: - prox for dual of L1 and L2 form

import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.utils import GradientOperator, GradientNorm
from pyparallelproj.algorithms import spdhg, osem_lm_emtv
import pymirc.image_operations as pi

from algorithms import spdhg_lm
from utils import  plot_lm_spdhg_results

from scipy.ndimage import gaussian_filter
import numpy as np
import argparse

from time import time

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',    help = 'counts to simulate',    default = 4e7, type = float)
parser.add_argument('--niter',     help = 'number of iterations',  default = 100, type = int)
parser.add_argument('--nsubsets',  help = 'number of subsets',     default = 112, type = int)
parser.add_argument('--fwhm_mm',   help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--seed',      help = 'seed for random generator', default = 1, type = int)
parser.add_argument('--beta',      help = 'TV weight',  default = 3e-2, type = float)
parser.add_argument('--prior',     help = 'prior',      default = 'TV', choices = ['TV','DTV'])
parser.add_argument('--rel_gamma', help = 'step size ratio',  default = 3, type = float)
parser.add_argument('--rho',       help = 'step size',        default = 0.999, type = float)
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
rel_gamma     = args.rel_gamma
rho           = args.rho

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,9]),
                                       nmodules             = np.array([28,4]))

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

# setup the ground truth image

lab = np.fromfile('data/xcat/georg_act_1.bin', dtype = np.float32).reshape((400,350,350)).swapaxes(0,2)
att_img = np.fromfile('data/xcat/georg_atn_1.bin', dtype = np.float32).reshape((400,350,350)).swapaxes(0,2)

img = np.zeros(lab.shape, dtype = np.float32)

# mycardium
img[lab == 1] = 2
# blood pool
img[lab == 2] = 1.2
# skin
img[lab == 3] = 1.
# fat
img[lab == 4] = 0.8
# liver
img[lab == 5] = 2.5
# gal bladder
img[lab == 6] = 2.
# lung
img[lab == 7] = 0.2
# spleen
img[lab == 8] = 1.5
# kidney outside
img[lab == 9] = 3.
# kidney inside
img[lab == 10] = 2.
# stomach
img[lab == 11] = 1.5
# cort ribs
img[lab == 12] = 2.
# cort. shoulder
img[lab == 13] = 2.
# cort. spine
img[lab == 14] = 2.
# spinal cord
img[lab == 15] = 1.
# spong. bone
img[lab == 16] = 1.5
# arteries
img[lab == 17] = 1.2
# venes
img[lab == 18] = 1.2

# soft tissues
img[lab == 21] = 1
img[lab == 22] = 1
img[lab == 23] = 1
# tachea
img[lab == 24] = 0

# soft tissue
img[lab == 42] = 1
# bowel cavities
img[lab == 43] = 0

# voxel size of xcat is ca 500/350 = 1.4mm -> extrapolate to 2mm voxels

img = pi.zoom3d(img, 1.4/2)
att_img = pi.zoom3d(att_img, 1.4/2)

# setup a test image
voxsize = np.array([2.,2.,2.])
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))
n0      = img.shape[0]
n1      = img.shape[1]

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# crop in z
img = img[...,50:(n2+50)]
att_img = att_img[...,50:(n2+50)]

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 21, tofbin_width = 27.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# create the attenuation sinogram
proj.set_tof(False)
attn_sino = np.exp(-proj.fwd_project(att_img))
proj.set_tof(True)

# generate the sensitivity sinogram
sens_sino = np.ones(proj.sino_params.nontof_shape, dtype = np.float32)

# power iterations to estimte norm of PET fwd operator
rimg  = np.random.rand(*img.shape).astype(np.float32)

for i in range(5):
  rimg_fwd = ppp.pet_fwd_model(rimg, proj, attn_sino, sens_sino, fwhm = fwhm)
  rimg     = ppp.pet_back_model(rimg_fwd, proj, attn_sino, sens_sino, fwhm = fwhm)

  pnsq     = np.linalg.norm(rimg)
  rimg    /= pnsq
  print(np.sqrt(pnsq))

del rimg_fwd

# scale the sensitivity and the image such that the norm of the PET fwd operator is approx 
# eual to the norm of the 3D gradient operator
sens_sino /= (np.sqrt(pnsq)/np.sqrt(12))

# forward project the image
img_fwd = ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, fwhm = fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.75*img_fwd.mean(), dtype = np.float32)
  
  em_sino = np.random.poisson(img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with sctter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.75*img_fwd.mean(), dtype = np.float32)

  em_sino = img_fwd + contam_sino

del img_fwd

#-------------------------------------------------------------------------------------

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

attn_list   = attn_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
sens_list   = sens_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2],0]
contam_list = contam_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]]

sens_img  = ppp.pet_back_model(np.ones(proj.sino_params.shape, dtype = np.float32), 
                               proj, attn_sino, sens_sino, fwhm = fwhm)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

G0 = GradientOperator()
if prior == 'DTV':
  grad_operator  = GradientOperator(joint_grad_field = G0.fwd(-(img**0.5)))
else:
  grad_operator  = GradientOperator()

grad_norm = GradientNorm(name = 'l2_l1')


xinit = osem_lm_emtv(events, attn_list, sens_list, contam_list, proj, sens_img, 1, 28, 
                     grad_operator = grad_operator, grad_norm = grad_norm,
                     fwhm = fwhm, verbose = True, beta = beta)

xinit[sens_img == 0] = 0
#yinit = 1 - (em_sino / (ppp.pet_fwd_model(xinit, proj, attn_sino, sens_sino, fwhm = fwhm) + contam_sino))

#-----------------------------------------------------------------------------------------------------

def calc_cost(x):
  cost = 0

  # split cost calculation over subsets to save memory
  for i, ss in enumerate(proj.subset_slices):
    exp = ppp.pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm) + contam_sino[ss]
    cost += (exp - em_sino[ss]*np.log(exp)).sum()

  if beta > 0:
    cost += beta*grad_norm.eval(grad_operator.fwd(x))

  return cost


def _cb(x, **kwargs):
  kwargs['t'].append(time())
  it = kwargs.get('iteration',0)

  if (it <= 10) or ((it % 10) == 0):
    kwargs['x_early'].append(x)
    kwargs['it_early'].append(it)
    np.save(f'it_{it:03}.npy',x)

  if 'cost' in kwargs:
    kwargs['cost'][it-1] = calc_cost(x)


#-----------------------------------------------------------------------------------------------------
norm = gaussian_filter(xinit.squeeze(),3).max()

#
##cbs_sino = {'x_early':[], 't':[]}
##x_sino = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
##               fwhm = fwhm, gamma = rel_gamma / norm, verbose = True, rho = rho, 
##               callback = _cb, callback_kwargs = cbs_sino,
##               xstart = xinit, ystart = yinit,
##               grad_operator = grad_operator, grad_norm = grad_norm, beta = beta)
##
#del yinit

#cost_sino2 = np.zeros(niter)
#cbs_sino2 = {'x_early':[], 't':[], 'it_early':[], 'cost' : cost_sino2}
#x_sino2 = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
#               fwhm = fwhm, gamma = rel_gamma / norm, verbose = True, rho = rho,
#               callback = _cb, callback_kwargs = cbs_sino2,
#               grad_operator = grad_operator, grad_norm = grad_norm, beta = beta)

cost_lm = np.zeros(niter)
cbs_lm = {'x_early':[], 't':[], 'it_early':[], 'cost' : cost_lm}
x_lm = spdhg_lm(events, attn_list, sens_list, contam_list, sens_img,
                proj, niter, nsubsets, x0 = xinit,
                fwhm = fwhm, gamma = rel_gamma / norm, verbose = True, rho = rho,
                callback = _cb, callback_kwargs = cbs_lm,
                grad_operator = grad_operator, grad_norm = grad_norm, beta = beta)


cost_emtv = np.zeros(niter)
cbs_emtv  = {'x_early':[], 't':[], 'it_early':[], 'cost' : cost_emtv}
x_emtv = osem_lm_emtv(events, attn_list, sens_list, contam_list, proj, sens_img, niter, 1, 
                      grad_operator = grad_operator, grad_norm = grad_norm, xstart = xinit,
                      callback = _cb, callback_kwargs = cbs_emtv,
                      fwhm = fwhm, verbose = True, beta = beta)


np.savez('debug.npz', x = x_lm, x_early = np.array(cbs_lm['x_early']), it = cbs_lm['it_early'], 
                      img = img, xinit = xinit, cost = cost_lm)

#-----------------------------------------------------------------------------------------------------

import pymirc.viewer as pv
vi = pv.ThreeAxisViewer(np.array(cbs_lm['x_early']), imshow_kwargs = {'vmax':15*counts/7e7,'vmin':0})
