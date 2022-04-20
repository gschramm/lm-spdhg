import numpy as np
import pymirc.viewer as pv
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

import pyparallelproj as ppp
from pyparallelproj.utils import GradientOperator, GradientNorm
from pyparallelproj.models import pet_fwd_model_lm, pet_back_model_lm

def load(path):
  vol = np.load(path)
  return vol 


def PSNR(x,y):
  MSE = ((x-y)**2).mean()
  return 20*np.log10(y.max()/np.sqrt(MSE))

#------------------------------------------------------------------------------

def calc_cost(x, proj, beta = 12, verbose = True):

  fwhm = 4.5 / (2.35*proj.voxsize)


  grad_operator = GradientOperator()
  grad_norm     = GradientNorm(name = 'l2_l1')
  
  cost = 0
  
  with h5py.File('data/dmi/lm_data.h5', 'r') as data:
    all_possible_events = data['all_xtals/xtal_ids'][:]
    all_sens  =  data['all_xtals/sens'][:]
    all_atten =  data['all_xtals/atten'][:]
  
    events = np.zeros((all_possible_events.shape[0], 5), dtype = all_possible_events.dtype)
    events[:,:-1] = all_possible_events
  
    # loop over all TOF bins
    for it in (np.arange(proj.ntofbins) - proj.ntofbins//2):
      events[:,-1]  = it
  
      contam = data[f'all_xtals/contam_{it}'][:]
      em     = data[f'all_xtals/emission_{it}'][:]
  
      if verbose:
        print(f'projecting TOF bin {it}')
      exp = pet_fwd_model_lm(x, proj, events, all_atten, all_sens, fwhm = fwhm) + contam
  
      cost += (exp - em*np.log(exp)).sum()
  
  if beta > 0:
    cost += beta*grad_norm.eval(grad_operator.fwd(x))

  return cost


#------------------------------------------------------------------------------

sdir = Path('data/dmi/NEMA_TV_beta_12_ss_136')

#------------------------------------------------------------------------------
# calculate the reference cost

voxsize   = np.array([2., 2., 2.], dtype = np.float32)
img_shape = (166,166,94)

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

#------------------------------------------------------------------------------

ref  = load(sdir / 'pdhg_lm_rho_1_it_2000.npy')

num_iter     = 50
PSNR_spdhg_1 = np.zeros(num_iter) 
PSNR_spdhg_4 = np.zeros(num_iter) 
PSNR_emtv_1  = np.zeros(num_iter) 
PSNR_emtv_8  = np.zeros(num_iter) 

spdhg_1 = np.zeros((num_iter,) + img_shape)
spdhg_4 = np.zeros((num_iter,) + img_shape)

emtv_1  = np.zeros((num_iter,) + img_shape)
emtv_8  = np.zeros((num_iter,) + img_shape)

for it in range(num_iter):
  spdhg_1[it,...] = load(sdir / f'spdhg_lm_rho_1_it_{(it+1):03}.npy')
  spdhg_4[it,...] = load(sdir / f'spdhg_lm_rho_4_it_{(it+1):03}.npy')

  emtv_1[it,...]  = load(sdir / f'emtv_1_it_{(it+1):03}.npy')
  emtv_8[it,...]  = load(sdir / f'emtv_8_it_{(it+1):03}.npy')

  PSNR_spdhg_1[it] = PSNR(spdhg_1[it,...],ref)
  PSNR_spdhg_4[it] = PSNR(spdhg_4[it,...],ref)

  PSNR_emtv_1[it]  = PSNR(emtv_1[it,...],ref)
  PSNR_emtv_8[it]  = PSNR(emtv_8[it,...],ref)

#ref_cost    = calc_cost(ref, proj)

#spdhg1_cost = calc_cost(spdhg_1[-1,...], proj)
#spdhg4_cost = calc_cost(spdhg_4[-1,...], proj)
#emtv1_cost  = calc_cost(emtv_1[-1,...], proj)
#emtv8_cost  = calc_cost(emtv_8[-1,...], proj)
#
#emtv34_cost = calc_cost(emtv_34[-1,...], proj)

##------------------------------------------------------------------------------

slz = 67
slx = 86

iterations = np.arange(1, num_iter+ 1)

ims  = {'vmin':0, 'vmax':0.41, 'cmap':plt.cm.Greys}
ims2 = {'vmin':-0.005, 'vmax':0.005, 'cmap':plt.cm.seismic}

fig, ax = plt.subplots(4,4, figsize = (10,10))
ax[0,0].plot(iterations,iterations)
ax[0,0].set_aspect(0.8)
ax[0,0].grid(ls = ':')
ax[0,0].set_xlabel('iteration')
ax[0,0].set_ylabel('relative cost')

ax[1,0].plot(iterations, PSNR_spdhg_1, color = plt.get_cmap("tab10")(1), label = "LM-SPDHG 136ss")
ax[1,0].plot(iterations, PSNR_emtv_1, color = plt.get_cmap("tab10")(2), label = "EMTV 1ss")
ax[1,0].plot(iterations, PSNR_emtv_8, color = plt.get_cmap("tab10")(3), label = "EMTV 8ss")
ax[1,0].set_aspect(74/166)
ax[1,0].set_ylim(0,65)
ax[1,0].grid(ls = ':')
ax[1,0].set_xlabel('iteration')
ax[1,0].set_ylabel('PSNR to reference')
ax[1,0].legend(fontsize = 'x-small')

ax[0,1].imshow(ref[...,slz], origin = 'lower', **ims)
ax[1,1].imshow(ref[slx,...,20:].T, **ims)
ax[0,1].set_title('reference LM-PDHG')

ax[0,2].imshow(spdhg_1[-1,...,slz], origin = 'lower', **ims)
ax[1,2].imshow(spdhg_1[-1,slx,...,20:].T, **ims)
ax[0,2].set_title(f'LM-SPDHG {num_iter}it./136ss.')

ax[2,2].imshow(spdhg_1[-1,...,slz] - ref[...,slz], origin = 'lower', **ims2)
ax[3,2].imshow(spdhg_1[-1,slx,...,20:].T - ref[slx,...,20:].T, **ims2)

ax[0,3].imshow(emtv_1[-1,...,slz], origin = 'lower', **ims)
ax[1,3].imshow(emtv_1[-1,slx,...,20:].T, **ims)
ax[0,3].set_title(f'EMTV {num_iter}it./1ss.')

ax[2,3].imshow(emtv_1[-1,...,slz] - ref[...,slz], origin = 'lower', **ims2)
ax[3,3].imshow(emtv_1[-1,slx,...,20:].T - ref[slx,...,20:].T, **ims2)

for axx in ax[:,1:].ravel():
   axx.set_axis_off()

for axx in ax[2:,0].ravel():
   axx.set_axis_off()

fig.tight_layout()
fig.show()

#iterations = np.arange(1, num_iter + 1)
#fig, ax = plt.subplots(2,2)
#ax[0,0].plot(iterations, PSNR_spdhg_1)
#ax[0,0].plot(iterations, PSNR_spdhg_8)
#ax[0,0].plot(iterations, PSNR_emtv_8)
#ax[0,0].plot(iterations, PSNR_emtv_34)
#
#ax[0,0].set_xlabel('iteration')
#ax[0,1].set_xlabel('PSNR to reference')
#
#fig.tight_layout()
#fig.show()
