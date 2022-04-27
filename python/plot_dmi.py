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

def calc_cost(x, proj, beta = 6, verbose = True):

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
      # TOF bin vs IDL framework must be reverted
      events[:,-1]  = -it
  
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
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

num_iter = 100
beta     = 6
sdir     = Path('data/dmi/NEMA_TV_beta_6_ss_224')

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
ref_file = sdir / 'pdhg_lm_rho_1_6_it_20000.npy'
ref_file_cost = ref_file.with_suffix('.txt')

ref  = load(ref_file)

if not ref_file_cost.exists():
  ref_cost = calc_cost(ref, proj, beta = beta)
  np.savetxt(ref_file_cost, [ref_cost])
else:
  ref_cost = float(np.loadtxt(ref_file_cost))

#------------------------------------------------------------------------------
init_file = sdir / 'init.npy'
init_file_cost = init_file.with_suffix('.txt')

init = load(init_file)

if not init_file_cost.exists():
  init_cost = calc_cost(init, proj, beta = beta)
  np.savetxt(init_file_cost, [init_cost])
else:
  init_cost = float(np.loadtxt(init_file_cost))


#------------------------------------------------------------------------------

PSNR_spdhg_1 = np.zeros(num_iter + 1) 
PSNR_emtv_1  = np.zeros(num_iter + 1) 
PSNR_emtv_28  = np.zeros(num_iter + 1) 

PSNR_spdhg_1[0]  = PSNR(init,ref)
PSNR_emtv_1[0]   = PSNR(init,ref)
PSNR_emtv_28[0]  = PSNR(init,ref)

cost_spdhg_1 = np.zeros(num_iter + 1) 
cost_emtv_1  = np.zeros(num_iter + 1) 
cost_emtv_28  = np.zeros(num_iter + 1) 

cost_spdhg_1[0] = init_cost
cost_emtv_1[0]  = init_cost
cost_emtv_28[0] = init_cost

spdhg_1 = np.zeros((num_iter,) + img_shape)
emtv_1  = np.zeros((num_iter,) + img_shape)
emtv_28  = np.zeros((num_iter,) + img_shape)


for it in range(num_iter):
  print(it+1,num_iter)

  spdhg_1_file = sdir / f'spdhg_lm_rho_1_beta_{beta}_it_{(it+1):03}.npy'
  emtv_1_file  = sdir / f'emtv_ss_1_beta_{beta}_it_{(it+1):03}.npy'
  emtv_28_file = sdir / f'emtv_ss_28_beta_{beta}_it_{(it+1):03}.npy'

  spdhg_1[it,...] = load(spdhg_1_file)
  emtv_1[it,...]  = load(emtv_1_file)
  emtv_28[it,...]  = load(emtv_28_file)

  PSNR_spdhg_1[it+1]  = PSNR(spdhg_1[it,...],ref)
  PSNR_emtv_1[it+1]   = PSNR(emtv_1[it,...],ref)
  PSNR_emtv_28[it+1]  = PSNR(emtv_28[it,...],ref)

  # calculate the cost
  spdhg_1_file_cost = spdhg_1_file.with_suffix('.txt')
  emtv_1_file_cost  = emtv_1_file.with_suffix('.txt')
  emtv_28_file_cost  = emtv_28_file.with_suffix('.txt')
  
  if not spdhg_1_file_cost.exists():
    cost_spdhg_1[it+1] = calc_cost(spdhg_1[it,...], proj, beta = beta)
    np.savetxt(spdhg_1_file_cost, [cost_spdhg_1[it+1]])
  else:
    cost_spdhg_1[it+1] = float(np.loadtxt(spdhg_1_file_cost))

  if not emtv_1_file_cost.exists():
    cost_emtv_1[it+1] = calc_cost(emtv_1[it,...], proj, beta = beta)
    np.savetxt(emtv_1_file_cost, [cost_emtv_1[it+1]])
  else:
    cost_emtv_1[it+1] = float(np.loadtxt(emtv_1_file_cost))

  if not emtv_28_file_cost.exists():
    cost_emtv_28[it+1] = calc_cost(emtv_28[it,...], proj, beta = beta)
    np.savetxt(emtv_28_file_cost, [cost_emtv_28[it+1]])
  else:
    cost_emtv_28[it+1] = float(np.loadtxt(emtv_28_file_cost))


#------------------------------------------------------------------------------

slz = 67
slx = 86

iterations = np.arange(num_iter+ 1)

ims  = {'vmin':0, 'vmax':0.5, 'cmap':plt.cm.Greys}
ims2 = {'vmin':-0.005, 'vmax':0.005, 'cmap':plt.cm.seismic}

fig, ax = plt.subplots(4,4, figsize = (10,9))
ax[0,0].set_title('reference LM-PDHG', fontsize = 'medium')
im00 = ax[0,0].imshow(ref[...,slz], origin = 'lower', **ims)
fig.colorbar(im00, ax = ax[0,0], fraction = 0.03, pad = 0.01, aspect = 30, location = 'bottom')
im10 = ax[1,0].imshow(ref[slx,...,20:].T, **ims)

ax[0,1].set_title(f'LM-SPDHG {num_iter}it/224ss', fontsize = 'medium')
ax[2,1].set_title(f'LM-SPDHG {num_iter}it/224ss - ref.', fontsize = 'medium')
im01 = ax[0,1].imshow(spdhg_1[-1,...,slz], origin = 'lower', **ims)
fig.colorbar(im01, ax = ax[0,1], fraction = 0.03, pad = 0.01, aspect = 30, location = 'bottom')
im11 = ax[1,1].imshow(spdhg_1[-1,slx,...,20:].T, **ims)
im21 = ax[2,1].imshow(spdhg_1[-1,...,slz] - ref[...,slz], origin = 'lower', **ims2)
cbar21 = fig.colorbar(im21, ax = ax[2,1], fraction = 0.03, pad = 0.01, aspect = 30, location = 'bottom')
cbar21.ax.locator_params(nbins = 2)
im31 = ax[3,1].imshow(spdhg_1[-1,slx,...,20:].T - ref[slx,...,20:].T, **ims2)

ax[0,2].set_title(f'EMTV {num_iter}it/1ss', fontsize = 'medium')
ax[2,2].set_title(f'EMTV {num_iter}it/1ss - ref.', fontsize = 'medium')
im02 = ax[0,2].imshow(emtv_1[-1,...,slz], origin = 'lower', **ims)
fig.colorbar(im02, ax = ax[0,2], fraction = 0.03, pad = 0.01, aspect = 30, location = 'bottom')
im12 = ax[1,2].imshow(emtv_1[-1,slx,...,20:].T, **ims)
im22 = ax[2,2].imshow(emtv_1[-1,...,slz] - ref[...,slz], origin = 'lower', **ims2)
cbar22 = fig.colorbar(im22, ax = ax[2,2], fraction = 0.03, pad = 0.01, aspect = 30, location = 'bottom')
cbar22.ax.locator_params(nbins = 2)
im32 = ax[3,2].imshow(emtv_1[-1,slx,...,20:].T - ref[slx,...,20:].T, **ims2)

ax[0,3].set_title(f'EMTV {num_iter}it/28ss', fontsize = 'medium')
ax[2,3].set_title(f'EMTV {num_iter}it/28ss - ref.', fontsize = 'medium')
im03 = ax[0,3].imshow(emtv_28[-1,...,slz], origin = 'lower', **ims)
fig.colorbar(im03, ax = ax[0,3], fraction = 0.03, pad = 0.01, aspect = 30, location = 'bottom')
im13 = ax[1,3].imshow(emtv_28[-1,slx,...,20:].T, **ims)
im23 = ax[2,3].imshow(emtv_28[-1,...,slz] - ref[...,slz], origin = 'lower', **ims2)
cbar23 = fig.colorbar(im23, ax = ax[2,3], fraction = 0.03, pad = 0.01, aspect = 30, location = 'bottom')
cbar23.ax.locator_params(nbins = 2)
im33 = ax[3,3].imshow(emtv_28[-1,slx,...,20:].T - ref[slx,...,20:].T, **ims2)

inds = np.where(cost_spdhg_1 != 0)

ax[2,0].semilogy(iterations[inds], (cost_emtv_1[inds] - ref_cost) / (init_cost - ref_cost), 
             color = plt.get_cmap("tab10")(2), label = "EMTV 1ss")
ax[2,0].semilogy(iterations[inds], (cost_emtv_28[inds] - ref_cost) / (init_cost - ref_cost), 
             color = plt.get_cmap("tab10")(4), label = "EMTV 28ss")
ax[2,0].semilogy(iterations[inds], (cost_spdhg_1[inds] - ref_cost) / (init_cost - ref_cost),
             color = plt.get_cmap("tab10")(1), label = "LM-SPDHG 224ss")
ax[2,0].grid(ls = ':')
ax[2,0].set_xlabel('iteration')
ax[2,0].set_ylabel('relative cost')
ax[2,0].legend(fontsize = 'x-small')

ax[3,0].plot(iterations, PSNR_emtv_1, color = plt.get_cmap("tab10")(2))
ax[3,0].plot(iterations, PSNR_emtv_28, color = plt.get_cmap("tab10")(4))
ax[3,0].plot(iterations, PSNR_spdhg_1, color = plt.get_cmap("tab10")(1))
ax[3,0].set_aspect(1.5*74/166)
ax[3,0].set_ylim(0,65)
ax[3,0].grid(ls = ':')
ax[3,0].set_xlabel('iteration')
ax[3,0].set_ylabel('PSNR to reference')

for axx in ax[:2,:].ravel():
   axx.set_axis_off()

for axx in ax[2:,1:].ravel():
   axx.set_axis_off()
fig.tight_layout()
fig.show()
