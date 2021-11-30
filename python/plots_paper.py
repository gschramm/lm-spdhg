import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path

mdir = Path('data/20211102_TMI')
nss = 112

for ofile in mdir.glob('brain2d_*.npz'):
  data = np.load(ofile)
  
  ref_recon       = data['ref_recon']
  cost_ref        = data['cost_ref']         
  
  x_sino          = data['x_sino']          
  x_sino_early    = data['x_sino_early']          
  cost_spdhg_sino = data['cost_spdhg_sino'] 
  psnr_spdhg_sino = data['psnr_spdhg_sino'] 
  
  x_lm            = data['x_lm']              
  x_lm_early      = data['x_lm_early']              
  cost_spdhg_lm   = data['cost_spdhg_lm']   
  psnr_spdhg_lm   = data['psnr_spdhg_lm']   
  
  x_emtv          = data['x_emtv']              
  x_emtv_early    = data['x_emtv_early']              
  cost_emtv       = data['cost_emtv']   
  psnr_emtv       = data['psnr_emtv']   
  
  nsubsets        = data['nsubsets']
  nsubsets_emtv   = data['nsubsets_emtv']
  
  gamma           = data['gamma']             
  img             = data['img']             
  c_0             = data['c_0']             
  
  #----------------------------------------------------------------------------------------------
  
  niter = cost_spdhg_sino.shape[1]
  
  # show the results
  vmax = 1.2*img.max()
  ncols = len(nsubsets)+ + len(nsubsets_emtv) + 1
  
  title_kwargs = {'fontweight':'bold', 'fontsize':'medium'}
  
  c_ref = cost_ref.min()
  n     = c_0 - c_ref
  ni    = np.arange(niter) + 1
  
  i = np.where(nsubsets == nss)[0][0]
  
  fig, ax = plt.subplots(2,4, figsize = (10,5))
  
  ax[0,0].semilogy(ni, (cost_spdhg_sino[i,:] - c_ref)/n, color = 'tab:blue')
  ax[0,0].semilogy(ni, (cost_spdhg_lm[i,:] - c_ref)/n, color = 'tab:orange')
  ax[1,0].plot(ni, psnr_spdhg_sino[i,:], color = 'tab:blue', label = f'SPDHG {nss}ss') 
  ax[1,0].plot(ni, psnr_spdhg_lm[i,:], color = 'tab:orange', label = f'LM-SPDHG {nss}ss') 
  
  ax[0,0].grid(ls = ':')
  ax[0,0].set_xlabel('iteration')
  ax[0,0].set_ylabel('relative cost')
  ax[1,0].grid(ls = ':')
  ax[1,0].set_xlabel('iteration')
  ax[1,0].set_ylabel('PSNR to reference')
  
  ax[0,0].set_ylim(1e-3,0.5)
  ax[1,0].set_ylim(28, 78)
  ax[1,0].legend()
  
  bfrac = 0.01
  
  im01 = ax[0,1].imshow(x_sino[i,...].squeeze()[:,19:-19], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  im11 = ax[1,1].imshow(x_lm[i,...].squeeze()[:,19:-19],   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  
  im02 = ax[0,2].imshow(x_sino[i,...].squeeze()[:,19:-19] - ref_recon.squeeze()[:,19:-19], 
                 vmin = -bfrac*vmax, vmax = bfrac*vmax, cmap = plt.cm.bwr)
  im12 = ax[1,2].imshow(x_lm[i,...].squeeze()[:,19:-19] - ref_recon.squeeze()[:,19:-19],   
                 vmin = -bfrac*vmax, vmax = bfrac*vmax, cmap = plt.cm.bwr)
  
  im03 = ax[0,3].imshow(ref_recon.squeeze()[:,19:-19], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  im13 = ax[1,3].imshow(img.squeeze()[:,19:-19], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  
  fig.colorbar(im01, ax = ax[0,1], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
  fig.colorbar(im11, ax = ax[1,1], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
  fig.colorbar(im02, ax = ax[0,2], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
  fig.colorbar(im12, ax = ax[1,2], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
  fig.colorbar(im03, ax = ax[0,3], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
  fig.colorbar(im13, ax = ax[1,3], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
  
  ax[0,1].set_title(f'SPDHG', **title_kwargs)
  ax[1,1].set_title(f'LM-SPDHG', **title_kwargs)
  ax[0,2].set_title(f'SPDHG - reference', **title_kwargs)
  ax[1,2].set_title(f'LM-SPDHG - reference', **title_kwargs)
  ax[0,3].set_title(f'reference PDHG', **title_kwargs)
  ax[1,3].set_title(f'ground truth', **title_kwargs)
  
  for axx in ax[:,1:].ravel():
    axx.set_axis_off()
  
  fig.tight_layout()
  fig.savefig(ofile.with_suffix('.png'))
  fig.show()
