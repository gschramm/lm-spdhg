import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path

mdir = Path('data/20211206_paper')
nss = 224

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
  
  title_kwargs = {'fontweight':'bold', 'fontsize':'small'}
  
  c_ref = cost_ref.min()
  n     = c_0 - c_ref
  ni    = np.arange(niter) + 1
  
  i = np.where(nsubsets == nss)[0][0]
  
  fig, ax = plt.subplots(2,5, figsize = (10,5))
  
  ax[0,0].semilogy(ni, (cost_spdhg_lm[i,:] - c_ref)/n, color = 'tab:orange')
  ax[1,0].plot(ni, psnr_spdhg_lm[i,:], color = 'tab:orange', label = f'LM-SPDHG {nss}ss') 
  
  for ig, nsse in enumerate(nsubsets_emtv):
    col = plt.get_cmap("tab10")(ig + 2)
    ax[0,0].semilogy(ni, (cost_emtv[ig,:] - c_ref)/n, '-', color = col)
    ax[1,0].plot(ni, psnr_emtv[ig,:], '-', ms = 5, color = col, label = f'EMTV {nsse}ss')
  
  ax[0,0].grid(ls = ':')
  ax[0,0].set_xlabel('iteration')
  ax[0,0].set_ylabel('relative cost')
  ax[1,0].grid(ls = ':')
  ax[1,0].set_xlabel('iteration')
  ax[1,0].set_ylabel('PSNR to reference')
  
  ax[0,0].set_ylim(1e-4,0.5)
  ax[1,0].set_ylim(28, 85)
  ax[1,0].legend(loc = 2, fontsize = 'x-small')
  
  bfrac = 0.05
  im01 = ax[0,1].imshow(x_lm[i,...].squeeze()[:,19:-19],   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  im11 = ax[1,1].imshow(x_lm[i,...].squeeze()[:,19:-19] - ref_recon.squeeze()[:,19:-19],   
                 vmin = -bfrac*vmax, vmax = bfrac*vmax, cmap = plt.cm.bwr)
  
  fig.colorbar(im01, ax = ax[0,1], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
  fig.colorbar(im11, ax = ax[1,1], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
  ax[0,1].set_title(f'LM-SPDHG {niter}it/{nss}ss', **title_kwargs)
  ax[1,1].set_title(f'LM-SPDHG {niter}it/{nss}ss - ref.', **title_kwargs)
  
  for ig, nsse in enumerate(nsubsets_emtv):
    im0 = ax[0,2+ig].imshow(x_emtv[ig,...].squeeze()[:,19:-19],   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
    im1 = ax[1,2+ig].imshow(x_emtv[ig,...].squeeze()[:,19:-19] - ref_recon.squeeze()[:,19:-19],   
                           vmin = -bfrac*vmax, vmax = bfrac*vmax, cmap = plt.cm.bwr)
    fig.colorbar(im0, ax = ax[0,2+ig], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
    fig.colorbar(im1, ax = ax[1,2+ig], fraction = 0.05, pad = 0, aspect = 30, location = 'bottom')
    ax[0,2+ig].set_title(f'EMTV {niter}it/{nsse}ss', **title_kwargs)
    ax[1,2+ig].set_title(f'EMTV {niter}it/{nsse}ss - ref.', **title_kwargs)
  
  for axx in ax[:,1:].ravel():
    axx.set_axis_off()
  
  fig.tight_layout()
  fig.savefig(ofile.parent / ofile.name.replace('.npz','_emtv.png'))
  fig.show()
