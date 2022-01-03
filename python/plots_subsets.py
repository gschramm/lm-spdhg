import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path

mdir = Path('data/20220103_paper')

for ofile in list(mdir.glob('brain2d_*niter_100.npz')):
  print(ofile.name)

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
  
  fig, ax = plt.subplots(2,1, figsize = (2.5,4), sharex = True)

  for i, nss in enumerate(nsubsets):
    nth = -1
    if psnr_spdhg_lm[i,:].max() >= 40:
      nth = np.where(psnr_spdhg_lm[i,:] >= 40)[0].min()
    print(f'{nss:3} {nth:3}')

    ax[0].semilogy(ni, (cost_spdhg_lm[i,:] - c_ref)/n, label = f'{nss:3} ss')
    ax[1].plot(ni, psnr_spdhg_lm[i,:]) 
    
  ax[0].grid(ls = ':')
  ax[0].set_ylabel('relative cost')
  ax[1].grid(ls = ':')
  ax[1].set_xlabel('iteration')
  ax[1].set_ylabel('PSNR to reference')
  
  ax[0].set_ylim(1e-5,0.5)
  ax[1].set_ylim(28, 90)
  ax[0].legend(ncol = 1, fontsize = 'small', loc = 0)
  
  fig.tight_layout()
  fig.savefig(ofile.parent / ofile.name.replace('.npz','_ss.pdf'))
  fig.show()
