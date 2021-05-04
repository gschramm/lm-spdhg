import os
import numpy as np
import matplotlib.pyplot as plt

def plot_lm_spdhg_results(base_str, subdir = 'data'):
  ofile    = os.path.join(subdir,f'{base_str}.npz')
  
  data = np.load(ofile)
  
  ref_recon       = data['ref_recon']
  cost_ref        = data['cost_ref']         
  cost_spdhg_sino = data['cost_spdhg_sino'] 
  psnr_spdhg_sino = data['psnr_spdhg_sino'] 
  cost_spdhg_lm   = data['cost_spdhg_lm']   
  psnr_spdhg_lm   = data['psnr_spdhg_lm']   
  x_sino          = data['x_sino']          
  x_lm            = data['x_lm']              
  gammas          = data['gammas']             
  img             = data['img']             
  c_0             = data['c_0']             
  
  #----------------------------------------------------------------------------------------------
  
  niter = cost_spdhg_sino.shape[1]
  
  # show the results
  vmax = 1.2*img.max()
  
  fig, ax = plt.subplots(2,len(gammas)+1, figsize = (3*(len(gammas)+1),6))
  for i,gam in enumerate(gammas):
    ax[0,i].imshow(x_sino[i,...].squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
    ax[1,i].imshow(x_lm[i,...].squeeze(),   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  
    ax[0,i].set_title(f'SPDHG {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
    ax[1,i].set_title(f'LM SPDHG {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  
  ax[0,-1].imshow(ref_recon.squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,-1].imshow(img.squeeze(),   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[0,-1].set_title(f'reference PDHG', fontsize = 'medium')
  ax[1,-1].set_title(f'ground truth', fontsize = 'medium')
  
  for axx in ax.ravel():
    axx.set_axis_off()
  
  fig.tight_layout()
  fig.show()
  
  c_ref = cost_ref.min()
  n     = c_0 - c_ref
  ni    = np.arange(niter) + 1
  
  fig2, ax2 = plt.subplots(1,2, figsize = (10,5))
  for ig, gam in enumerate(gammas):
    col = plt.get_cmap("tab10")(ig)
    ax2[0].semilogy(ni, (cost_spdhg_sino[ig,:] - c_ref)/n, '-.', ms = 5,
                 label = f'SINO {gam:.2e}',color = col)
    ax2[0].semilogy(ni, (cost_spdhg_lm[ig,:] - c_ref)/n, '-v', ms = 5, 
                 label = f'LM {gam:.2e}', color = col)
    ax2[1].plot(ni, psnr_spdhg_sino[ig,:], '-.', ms = 5, color = col)
    ax2[1].plot(ni, psnr_spdhg_lm[ig,:], '-v', ms = 5, color = col)
  
  ax2[0].grid(ls = ':')
  ax2[0].set_xlabel('iteration')
  ax2[0].set_ylabel('relative cost')
  ax2[1].grid(ls = ':')
  ax2[1].set_xlabel('iteration')
  ax2[1].set_ylabel('PSNR to reference')
  
  handles, labels = ax2[0].get_legend_handles_labels()
  fig2.legend(handles, labels, loc='upper center', ncol = len(gammas), fontsize = 'small')
  
  fig2.tight_layout()
  fig2.show()
  
  # save the figures
  fig.savefig(os.path.join(subdir,f'{base_str}.png'))
  fig2.savefig(os.path.join(subdir,f'{base_str}_metrics.pdf'))
  fig2.savefig(os.path.join(subdir,f'{base_str}_metrics.png'))

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  from glob import glob
  fnames = glob(os.path.join('data','brain2d*.npz'))

  for i, fname in enumerate(fnames):
    base_str = os.path.splitext(os.path.basename(fname))[0]
    if not os.path.exists(os.path.join('data',f'{base_str}.png')):
      plot_lm_spdhg_results(base_str, subdir = 'data')
