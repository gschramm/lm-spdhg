import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def strip_axes(axx):
  axx.set_xticks([])
  axx.set_yticks([])
  axx.spines['top'].set_visible(False)
  axx.spines['right'].set_visible(False)
  axx.spines['bottom'].set_visible(False)
  axx.spines['left'].set_visible(False)
 
#--------------------------------------------------------------------------------------- 

def plot_mean_std(prior = 'TV'):

  if prior == 'TV':
    fnames = list(pathlib.Path('data/20211009_MIC_DTV_TV').glob('brain2d_counts_1.0E+06_seed_*_beta_3.0E-03_prior_TV_niter_ref_10000_fwhm_4.5_4.5_niter_100_nsub_56.npz'))
  elif prior == 'DTV':
    fnames = list(pathlib.Path('data/20211009_MIC_DTV_TV').glob('brain2d_counts_1.0E+06_seed_*_beta_3.0E-02_prior_DTV_niter_ref_10000_fwhm_4.5_4.5_niter_100_nsub_56.npz'))
  
  n_real = len(fnames)
  x_lm   = np.zeros((n_real,128,128))
  x_sino = np.zeros((n_real,128,128))
  x_ref  = np.zeros((n_real,128,128))
  
  for ir, fname in enumerate(fnames):
    print(ir+1,fname)
    data = np.load(fname)
    
    x_ref[ir,...]  = data['ref_recon'].squeeze()
    x_sino[ir,...] = data['x_sino'].squeeze()          
    x_lm[ir,...]   = data['x_lm'].squeeze()              
  
    if ir == 0:
      img = data['img']             
  
  #------------------------------------------------------------------------------------------------------------
  
  vmax1 = 1.1*img.max()
  vmax2 = 0.4*img.max()
   
  fig, ax = plt.subplots(3,5, figsize = (14,9))
  ax[0,0].imshow(x_ref[0,...][5:-5,15:-15],  vmin = 0, vmax = vmax1, cmap = plt.cm.Greys)
  ax[0,1].imshow(x_ref.mean(0)[5:-5,15:-15], vmin = 0, vmax = vmax1, cmap = plt.cm.Greys)
  ax[0,2].imshow(0*x_ref.mean(0)[5:-5,15:-15], vmin = 0, vmax = vmax1, cmap = plt.cm.Greys)
  ax[0,3].imshow(x_ref.std(0)[5:-5,15:-15],  vmin = 0, vmax = vmax2, cmap = plt.cm.Greys)
  ax[0,4].imshow(0*x_ref.mean(0)[5:-5,15:-15], vmin = 0, vmax = vmax1, cmap = plt.cm.Greys)
  
  ax[1,0].imshow(x_sino[0,...][5:-5,15:-15], vmin = 0, vmax = vmax1, cmap = plt.cm.Greys)
  ax[1,1].imshow(x_sino.mean(0)[5:-5,15:-15],vmin = 0, vmax = vmax1, cmap = plt.cm.Greys)
  ax[1,2].imshow(x_sino.mean(0)[5:-5,15:-15] - x_ref.mean(0)[5:-5,15:-15], vmin = -0.1*vmax1, vmax = 0.1*vmax1, cmap = plt.cm.bwr)
  ax[1,3].imshow(x_sino.std(0)[5:-5,15:-15], vmin = 0, vmax = vmax2, cmap = plt.cm.Greys)
  ax[1,4].imshow(x_sino.std(0)[5:-5,15:-15] - x_ref.std(0)[5:-5,15:-15], vmin = -0.1*vmax2, vmax = 0.1*vmax2, cmap = plt.cm.bwr)
  
  ax[2,0].imshow(x_lm[0,...][5:-5,15:-15],   vmin = 0, vmax = vmax1, cmap = plt.cm.Greys)
  ax[2,1].imshow(x_lm.mean(0)[5:-5,15:-15],  vmin = 0, vmax = vmax1, cmap = plt.cm.Greys)
  ax[2,2].imshow(x_lm.mean(0)[5:-5,15:-15] - x_ref.mean(0)[5:-5,15:-15], vmin = -0.1*vmax1, vmax = 0.1*vmax1, cmap = plt.cm.bwr)
  ax[2,3].imshow(x_lm.std(0)[5:-5,15:-15],   vmin = 0, vmax = vmax2, cmap = plt.cm.Greys)
  ax[2,4].imshow(x_lm.std(0)[5:-5,15:-15] - x_ref.std(0)[5:-5,15:-15], vmin = -0.1*vmax2, vmax = 0.1*vmax2, cmap = plt.cm.bwr)
  
  ax[0,0].set_ylabel('PDHG 10000 it')
  ax[1,0].set_ylabel('SPDHG 100 it, 56 ss')
  ax[2,0].set_ylabel('LM-SPDHG 100 it, 56 ss')
  
  ax[0,0].set_title('1st noise real.')
  ax[0,1].set_title(f'mean image {n_real} n.r.')
  ax[0,2].set_title('mean image - mean(PHDG)')
  ax[0,3].set_title(f'std. dev. image {n_real} n.r.')
  ax[0,4].set_title('std.dev. image - std.dev(PHDG)')
 
  for axx in ax.ravel(): strip_axes(axx) 
 
  fig.tight_layout()
  fig.show()

  return fig

#--------------------------------------------------------------------------------------- 
def plot_lm_spdhg_results_mic(ofile):
  data = np.load(ofile)
  
  ref_recon       = data['ref_recon']
  cost_ref        = data['cost_ref']         
  x_sino          = data['x_sino']          
  cost_spdhg_sino = data['cost_spdhg_sino'] 
  psnr_spdhg_sino = data['psnr_spdhg_sino'] 
  x_lm            = data['x_lm']              
  cost_spdhg_lm   = data['cost_spdhg_lm']   
  psnr_spdhg_lm   = data['psnr_spdhg_lm']   
  gammas          = data['gammas']             
  img             = data['img']             
  c_0             = data['c_0']             
  
  #----------------------------------------------------------------------------------------------
  
  niter = cost_spdhg_sino.shape[1]
  
  c_ref = cost_ref.min()
  n     = c_0 - c_ref
  ni    = np.arange(niter) + 1

  vmax = 1.1*img.max()

  fig = plt.figure(constrained_layout = False, figsize = (12,6))
  gs = fig.add_gridspec(3, 6)
  ax0 = fig.add_subplot(gs[0, 0])
  ax1 = fig.add_subplot(gs[0, 1])
  ax2 = fig.add_subplot(gs[1, 0])
  ax3 = fig.add_subplot(gs[1, 1])
  ax4 = fig.add_subplot(gs[2, 0])
  ax5 = fig.add_subplot(gs[2, 1])
  ax6 = fig.add_subplot(gs[:, 2:4])
  ax7 = fig.add_subplot(gs[:, 4:6])
  
  ax0.imshow(img.squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax1.imshow(ref_recon.squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax2.imshow(x_sino[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax3.imshow(x_lm[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax4.imshow(x_sino[0,...].squeeze()[5:-5,15:-15] - ref_recon.squeeze()[5:-5,15:-15], 
             vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)
  ax5.imshow(x_lm[0,...].squeeze()[5:-5,15:-15] - ref_recon.squeeze()[5:-5,15:-15], 
             vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)

  ax0.set_title('ground truth', fontsize = 'medium')
  ax1.set_title('PDHG 10000 it', fontsize = 'medium')
  ax2.set_title('SPDHG 100/56', fontsize = 'medium')
  ax3.set_title('LM-SPDHG 100/56', fontsize = 'medium')
  ax4.set_title('SPDHG - PDHG', fontsize = 'medium')
  ax5.set_title('LM-SPDHG - PDHG', fontsize = 'medium')
 
  ax6.semilogy(ni, (cost_spdhg_sino[0,:] - c_ref)/n, '-o', ms = 4, lw = 3, 
               label = f'SPDHG', color = 'tab:blue')
  ax6.semilogy(ni, (cost_spdhg_lm[0,:] - c_ref)/n, '-v', ms = 3, lw = 1.5, 
               label = f'LM-SPDHG', color = 'tab:orange')
  ax7.plot(ni, psnr_spdhg_sino[0,:], '-o', ms = 4, color = 'tab:blue', lw = 3)
  ax7.plot(ni, psnr_spdhg_lm[0,:], '-v', ms = 3, color = 'tab:orange', lw = 1.5)

  ax6.legend()

  ax6.grid(ls = ':')
  ax6.set_xlabel('iteration')
  ax6.set_ylabel('relative cost')
  ax7.grid(ls = ':')
  ax7.set_xlabel('iteration')
  ax7.set_ylabel('PSNR to reference')

  strip_axes(ax0)
  strip_axes(ax1)
  strip_axes(ax2)
  strip_axes(ax3)
  strip_axes(ax4)
  strip_axes(ax5)

  fig.tight_layout()
  fig.show()

  return fig
  
#--------------------------------------------------------------------------------------- 

def plot_conv_vs_subsets(nsubsets = [28,56,112,224],
                         prior    = 'DTV',
                         beta     = '3.0E-02'):

  fnames = [pathlib.Path('data/20211009_MIC_DTV_TV') / f'brain2d_counts_1.0E+06_seed_1_beta_{beta}_prior_{prior}_niter_ref_10000_fwhm_4.5_4.5_niter_100_nsub_{x}.npz' for x in nsubsets]

  fig, ax = plt.subplots(1,2, figsize = (10,4))
  axins1 = inset_axes(ax[1], width="45%", height="45%", loc=4, borderpad=1)

  for i, fname in enumerate(fnames):
    print(fname)
    data = np.load(fname)
    cost_ref        = data['cost_ref']         
    cost_spdhg_sino = data['cost_spdhg_sino'] 
    psnr_spdhg_sino = data['psnr_spdhg_sino'] 
    cost_spdhg_lm   = data['cost_spdhg_lm']   
    psnr_spdhg_lm   = data['psnr_spdhg_lm']   
    c_0             = data['c_0']             
    
    #----------------------------------------------------------------------------------------------
    
    niter = cost_spdhg_sino.shape[1]
    
    c_ref = cost_ref.min()
    n     = c_0 - c_ref
    ni    = np.arange(niter) + 1

    color = plt.get_cmap("tab10")(i)

    ax[0].semilogy(ni, (cost_spdhg_sino[0,:] - c_ref)/n, '-o', ms = 3, lw = 1.5, 
                 label = f'SPDHG {nsubsets[i]} ss', color = color)
    ax[0].semilogy(ni, (cost_spdhg_lm[0,:] - c_ref)/n, '-v', ms = 3, lw = 1.5, 
                 label = f'LM-SPDHG {nsubsets[i]} ss', color = color)
    ax[1].plot(ni, psnr_spdhg_sino[0,:], '-o', ms = 3, lw = 1.5, color = color)
    ax[1].plot(ni, psnr_spdhg_lm[0,:], '-v', ms = 3, lw = 1.5, color = color)

    axins1.plot(ni[3:25], psnr_spdhg_sino[0,:][3:25], '-o', ms = 3, lw = 1.5, color = color)
    axins1.plot(ni[3:25], psnr_spdhg_lm[0,:][3:25], '-v', ms = 3, lw = 1.5, color = color)


  ax[1].indicate_inset_zoom(axins1, edgecolor="black")
  ax[0].legend()

  for axx in ax: axx.grid(ls = ':')
  axins1.grid(ls = ':')

  ax[0].set_xlabel('iteration')
  ax[0].set_ylabel('relative cost')
  ax[1].set_xlabel('iteration')
  ax[1].set_ylabel('PSNR to reference')

  fig.tight_layout()
  fig.show()

#--------------------------------------------------------------------------------------- 

def plot_early_stopping():
  fname = 'data/brain2d_counts_1.0E+06_seed_1_beta_3.0E-03_prior_TV_niter_ref_10000_fwhm_4.5_4.5_niter_100_nsub_224.npz'

  data = np.load(fname)

  ref_recon = data['ref_recon']
  img       = data['img']

  x_early1_sino = data['x_early1_sino']
  x_early2_sino = data['x_early2_sino'] 
  x_early3_sino = data['x_early3_sino'] 
  x_early4_sino = data['x_early4_sino'] 

  x_early1_lm =   data['x_early1_lm']
  x_early2_lm =   data['x_early2_lm']
  x_early3_lm =   data['x_early3_lm']
  x_early4_lm =   data['x_early4_lm']

  x_sino = data['x_sino']          
  x_lm   = data['x_lm']          

  vmax = 1.1*img.max()

  fig, ax = plt.subplots(2,6, figsize = (6*2.3, 2*2.5))

  ax[0,0].imshow(img.squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,0].imshow(ref_recon.squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)

  ax[0,1].imshow(x_early1_sino[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[0,2].imshow(x_early2_sino[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[0,3].imshow(x_early3_sino[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[0,4].imshow(x_early4_sino[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[0,5].imshow(x_sino.squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)

  ax[1,1].imshow(x_early1_lm[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,2].imshow(x_early2_lm[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,3].imshow(x_early3_lm[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,4].imshow(x_early4_lm[0,...].squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,5].imshow(x_lm.squeeze()[5:-5,15:-15], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)

  for axx in ax.ravel(): strip_axes(axx)

  ax[0,0].set_title('ground truth', fontsize = 'small')
  ax[1,0].set_title('PDHG 10000 it', fontsize = 'small')
  ax[0,1].set_title('SPDHG 1 it / 224 ss', fontsize = 'small')
  ax[1,1].set_title('LM-SPDHG 1 it / 224 ss', fontsize = 'small')
  ax[0,2].set_title('SPDHG 2 it / 224 ss', fontsize = 'small')
  ax[1,2].set_title('LM-SPDHG 2 it / 224 ss', fontsize = 'small')
  ax[0,3].set_title('SPDHG 5 it / 224 ss', fontsize = 'small')
  ax[1,3].set_title('LM-SPDHG 5 it / 224 ss', fontsize = 'small')
  ax[0,4].set_title('SPDHG 10 it / 224 ss', fontsize = 'small')
  ax[1,4].set_title('LM-SPDHG 10 it / 224 ss', fontsize = 'small')
  ax[0,5].set_title('SPDHG 100 it / 224 ss', fontsize = 'small')
  ax[1,5].set_title('LM-SPDHG 100 it / 224 ss', fontsize = 'small')

  fig.tight_layout()
  fig.savefig('stopping_early.png')
  fig.show()

#--------------------------------------------------------------------------------------- 

if __name__ == '__main__':
  print()

  #fnames = pathlib.Path('data/20211009_MIC_DTV_TV').glob('brain2d_*.npz')

  #for fname in fnames: 
  #  fig = plot_lm_spdhg_results_mic(fname)
  #  fig.savefig(fname.with_suffix('.png'))

  #f1 = plot_mean_std(prior = 'TV')
  #f2 = plot_mean_std(prior = 'DTV')


