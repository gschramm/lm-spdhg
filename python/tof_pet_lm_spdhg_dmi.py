import pyparallelproj as ppp
from pyparallelproj.utils import GradientOperator, GradientNorm
from pyparallelproj.algorithms import spdhg, osem_lm_emtv
from pyparallelproj.models import pet_fwd_model_lm, pet_back_model_lm
from algorithms import spdhg_lm
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pathlib import Path
from utils import count_event_multiplicity

def _cb(x, **kwargs):
  it = kwargs.get('iteration',0)
  prefix = kwargs.get('prefix','')

  if (not prefix.startswith('pdhg')) or ((it % 100) == 0) or (it < 10):
    np.save(f'data/dmi/{prefix}_it_{it:03}.npy',x)

#--------------------------------------------------------------------------------------------

voxsize   = np.array([2., 2., 2.], dtype = np.float32)
img_shape = (166,166,94)
verbose   = True

# FHHM for resolution model in voxel
fwhm  = 4.5 / (2.35*voxsize)

# prior strength
beta = 6

niter    = 100
nsubsets = 224

np.random.seed(1)

#--------------------------------------------------------------------------------------------
# setup scanner and projector

# define the 4 ring DMI geometry
scanner = ppp.RegularPolygonPETScanner(
               R                    = 0.5*(744.1 + 2*8.51),
               ncrystals_per_module = np.array([16,9]),
               crystal_size         = np.array([4.03125,5.31556]),
               nmodules             = np.array([34,4]),
               module_gap_axial     = 2.8)


# define sinogram parameter - also needed to setup the LM projector

# speed of light in mm/ns
speed_of_light = 300.

# time resolution FWHM in ns
time_res_FWHM = 0.385

# sigma TOF in mm
sigma_tof = (speed_of_light/2) * (time_res_FWHM/2.355)

# the TOF bin width in mm is 13*0.01302ns times the speed of light (300mm/ns) divided by two
sino_params = ppp.PETSinogramParameters(scanner, rtrim = 65, ntofbins = 29, 
                                        tofbin_width = 13*0.01302*speed_of_light/2)

# define the projector
proj = ppp.SinogramProjector(scanner, sino_params, img_shape,
                             voxsize = voxsize, tof = True, 
                             sigma_tof = sigma_tof, n_sigmas = 3.)


#--------------------------------------------------------------------------------------------
# calculate sensitivity image

if verbose:
  print('Reading sens / atten "sinogram"')

with h5py.File('data/dmi/lm_data.h5', 'r') as data:
  all_possible_events = data['all_xtals/xtal_ids'][:]
  all_sens  =  data['all_xtals/sens'][:]
  all_atten =  data['all_xtals/atten'][:]

# calculate the sensivity image
# TODO: use TOF backprojector for sens_img calculation
if verbose:
  print('Calculating sensitivity image')
ones = np.ones(all_possible_events.shape[0], dtype = np.float32)

# create TOF sens image
sens_img = np.zeros(img_shape, dtype = np.float32)
events = np.zeros((all_possible_events.shape[0], 5), dtype = all_possible_events.dtype)
events[:,:-1] = all_possible_events

# loop over all TOF bins
for it in (np.arange(proj.ntofbins) - proj.ntofbins//2):
  print(it)
  events[:,-1]  = it
  sens_img += pet_back_model_lm(ones, proj, events, all_atten, all_sens, fwhm = fwhm) 
  
np.save(f'data/dmi/sens_img.npy',sens_img)
# delete all event variables that are not needed any more
del ones
del all_possible_events
del all_sens
del all_atten

#--------------------------------------------------------------------------------------------

# read the actual LM data and the correction lists

# read the LM data
if verbose:
  print('Reading LM data')

with h5py.File('data/dmi/lm_data.h5', 'r') as data:
  sens_list   =  data['correction_lists/sens'][:]
  atten_list  =  data['correction_lists/atten'][:]
  contam_list =  data['correction_lists/contam'][:]
  LM_file     =  Path(data['header/listfile'][0].decode("utf-8"))

with h5py.File(LM_file, 'r') as data:
  events = data['MiceList/TofCoinc'][:]

# swap axial and trans-axial crystals IDs
events = events[:,[1,0,3,2,4]]

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed 

events[:,-1] = -(events[:,-1]//13)

nevents = events.shape[0]

## shuffle events since events come semi sorted
if verbose: 
  print('shuffling LM data')
ie = np.arange(nevents)
np.random.shuffle(ie)
events = events[ie,:]
sens_list   = sens_list[ie]
atten_list  = atten_list[ie]  
contam_list = contam_list[ie]

# calculate the events multiplicity
if verbose: print('Calculating event count')
mu = count_event_multiplicity(events, use_gpu_if_possible = True)

# back project ones
if verbose: print('backprojecting LM data')
b_img = proj.back_project_lm(np.ones(nevents, dtype = np.float32), events)
np.save(f'data/dmi/b_img.npy',b_img)


#--------------------------------------------------------------------------------------------
grad_operator = GradientOperator()
grad_norm     = GradientNorm(name = 'l2_l1')

# run EM-TV
if verbose: print('running init EM-TV')
xinit = osem_lm_emtv(events, atten_list, sens_list, contam_list, proj, sens_img, 1, 34, 
                     grad_operator = grad_operator, grad_norm = grad_norm,
                     fwhm = fwhm, verbose = True, beta = beta)

np.save(f'data/dmi/init.npy',xinit)

norm = gaussian_filter(xinit.squeeze(),3).max()

xpdhg = spdhg_lm(events, atten_list, sens_list, contam_list, sens_img,
                 proj, 20000, 1, x0 = xinit,
                 fwhm = fwhm, gamma = 30 / norm, verbose = True, rho = 1,
                 callback = _cb, callback_kwargs = {'prefix':f'pdhg_lm_rho_1_{beta}'},
                 grad_operator = grad_operator, grad_norm = grad_norm, beta = beta)


xspdhg1 = spdhg_lm(events, atten_list, sens_list, contam_list, sens_img,
                   proj, niter, nsubsets, x0 = xinit,
                   fwhm = fwhm, gamma = 30 / norm, verbose = True, rho = 0.999,
                   callback = _cb, callback_kwargs = {'prefix':f'spdhg_lm_rho_1_beta_{beta}'},
                   grad_operator = grad_operator, grad_norm = grad_norm, beta = beta)

xemtv1 = osem_lm_emtv(events, atten_list, sens_list, contam_list, proj, sens_img, niter, 1, 
                      grad_operator = grad_operator, grad_norm = grad_norm, xstart = xinit,
                      callback = _cb, callback_kwargs = {'prefix':f'emtv_ss_1_beta_{beta}'},
                      fwhm = fwhm, verbose = True, beta = beta)

xemtv28 = osem_lm_emtv(events, atten_list, sens_list, contam_list, proj, sens_img, niter, 28, 
                       grad_operator = grad_operator, grad_norm = grad_norm, xstart = xinit,
                       callback = _cb, callback_kwargs = {'prefix':f'emtv_ss_28_beta_{beta}'},
                       fwhm = fwhm, verbose = True, beta = beta)
