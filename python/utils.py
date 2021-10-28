import os
import numpy as np
import matplotlib.pyplot as plt
import pkgutil
import math

def plot_lm_spdhg_results(ofile):
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
  
  fig, ax = plt.subplots(4, ncols, figsize = (2.5*ncols,2.5*4))
  for i,nss in enumerate(nsubsets):
    ax[0,i].imshow(x_sino[i,...].squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
    ax[1,i].imshow(x_lm[i,...].squeeze(),   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  
    ax[2,i].imshow(x_sino[i,...].squeeze() - ref_recon.squeeze(), 
                   vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)
    ax[3,i].imshow(x_lm[i,...].squeeze() - ref_recon.squeeze(),   
                   vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)
  
    ax[0,i].set_title(f'SPDHG {nss}ss', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
    ax[1,i].set_title(f'LM-SPDHG {nss}ss', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
    ax[2,i].set_title(f'bias SPDHG {nss}ss', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
    ax[3,i].set_title(f'bias LM-SPDHG {nss}ss', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  
  for i,nss in enumerate(nsubsets_emtv):
    col = i+len(nsubsets)
    ax[0,col].imshow(x_emtv[i,...].squeeze(), vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
    ax[2,col].imshow(x_emtv[i,...].squeeze() - ref_recon.squeeze(), 
                     vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)
  
    ax[0,col].set_title(f'EMTV {nss}ss', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
    ax[2,col].set_title(f'bias EMTV {nss}ss', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  
  
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
  for ig, nss in enumerate(nsubsets):
    col = plt.get_cmap("tab10")(ig)
    ax2[0].semilogy(ni, (cost_spdhg_sino[ig,:] - c_ref)/n, '-.', ms = 5,
                 label = f'SPDHG {nss}ss',color = col)
    ax2[0].semilogy(ni, (cost_spdhg_lm[ig,:] - c_ref)/n, '-v', ms = 5, 
                 label = f'LM-SPDHG {nss}ss', color = col)
    ax2[1].plot(ni, psnr_spdhg_sino[ig,:], '-.', ms = 5, color = col)
    ax2[1].plot(ni, psnr_spdhg_lm[ig,:], '-v', ms = 5, color = col)
  
  for ig, nss in enumerate(nsubsets_emtv):
    col = plt.get_cmap("tab10")(ig + len(nsubsets))
    ax2[0].semilogy(ni, (cost_emtv[ig,:] - c_ref)/n, '-', ms = 5, lw = 1,
                 label = f'EMTV {nss}ss',color = col)
    ax2[1].plot(ni, psnr_emtv[ig,:], '-', ms = 5, color = col, lw = 1)
  
  
  ax2[0].grid(ls = ':')
  ax2[0].set_xlabel('iteration')
  ax2[0].set_ylabel('relative cost')
  ax2[1].grid(ls = ':')
  ax2[1].set_xlabel('iteration')
  ax2[1].set_ylabel('PSNR to reference')
  
  handles, labels = ax2[0].get_legend_handles_labels()
  fig2.legend(handles, labels, loc='upper center', ncol = len(nsubsets) + math.ceil(len(nsubsets_emtv)/2), 
              fontsize = 'small')
  fig2.tight_layout()
  fig2.show()
  
  base_str = os.path.splitext(ofile)[0]
  
  # save the figures
  fig.savefig(base_str + '.png')
  fig2.savefig(base_str + '_metrics.pdf')
  fig2.savefig(base_str + '_metrics.png')

#----------------------------------------------------------------------------------------------------

def count_event_multiplicity(events, use_gpu_if_possible = True):
  """ Count the multiplicity of events in an LM file

  Parameters
  ----------

  events : 2D numpy array
    of LM events of shape (n_events, 5) where the second axis encodes the event 
    (e.g. detectors numbers and TOF bins)

  use_gpu_if_possible : bool
    whether to use a GPU and cupy if possible (default True)
  """

  cupy_available = False
  if pkgutil.find_loader("cupy"):
    cupy_available = True

  if (cupy_available and use_gpu_if_possible):
    import cupy
    from utils_cupy import cupy_unique_axis0

    events_d = cupy.array(events)
    tmp_d    = cupy_unique_axis0(events_d, return_counts = True, return_inverse = True)
    mu_d     = tmp_d[2][tmp_d[1]]
    mu       = cupy.asnumpy(mu_d)

  else:
    tmp = np.unique(events, axis = 0, return_counts = True, return_inverse = True)
    mu  = tmp[2][tmp[1]]

  return mu


#----------------------------------------------------------------------------------------------------
