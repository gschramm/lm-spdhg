import os
import numpy as np
import matplotlib.pyplot as plt

base_str = 'brain2d_counts_1.0E+06_beta_2.0E-03_niter_5000_100_nsub_56'
subdir   = os.path.join('..','..','python','data','20210505')
precond  = False

ofile    = os.path.join(subdir,f'{base_str}.npz')

data = np.load(ofile)

igs = slice(1,4,None)

early = ''

ref_recon       = data['ref_recon']
cost_ref        = data['cost_ref']         
x_sino          = data[f'x{early}_sino'][igs,...]          
cost_spdhg_sino = data['cost_spdhg_sino'][igs,...] 
psnr_spdhg_sino = data['psnr_spdhg_sino'][igs,...] 
if precond:
  x_lm            = data[f'x{early}_lm'][igs,...]              
  cost_spdhg_lm   = data['cost_spdhg_lm'][igs,...]   
  psnr_spdhg_lm   = data['psnr_spdhg_lm'][igs,...]   
else:
  x_lm            = data[f'x{early}_lm2'][igs,...]              
  cost_spdhg_lm   = data['cost_spdhg_lm2'][igs,...]   
  psnr_spdhg_lm   = data['psnr_spdhg_lm2'][igs,...]   
gammas          = data['gammas'][igs,...]             
img             = data['img']             
c_0             = data['c_0']             

#----------------------------------------------------------------------------------------------

niter = cost_spdhg_sino.shape[1]

# show the results
vmax = 1.2*img.max()

sl = (slice(8,-8,None),slice(18,-18,None))

fig, ax = plt.subplots(4,len(gammas)+1, figsize = (0.8*3*(len(gammas)+1),12))
for i,gam in enumerate(gammas):
  ax[0,i].imshow(x_sino[i,...].squeeze()[sl], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  ax[1,i].imshow(x_lm[i,...].squeeze()[sl],   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)

  ax[2,i].imshow(x_sino[i,...].squeeze()[sl] - ref_recon.squeeze()[sl], 
                 vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)
  ax[3,i].imshow(x_lm[i,...].squeeze()[sl] - ref_recon.squeeze()[sl],   
                 vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)

  ax[0,i].set_title(f'SINO {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  ax[1,i].set_title(f'LM   {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  ax[2,i].set_title(f'diff to ref SINO {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  ax[3,i].set_title(f'diff to ref LM   {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))

ax[0,-1].imshow(ref_recon.squeeze()[sl], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
ax[1,-1].imshow(img.squeeze()[sl],       vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
ax[0,-1].set_title(f'reference PDHG', fontsize = 'medium')
ax[1,-1].set_title(f'ground truth', fontsize = 'medium')

for axx in ax.ravel():
  axx.set_axis_off()

fig.tight_layout()
fig.show()

c_ref = cost_ref.min()
n     = c_0 - c_ref
ni    = np.arange(niter) + 1

fig2, ax2 = plt.subplots(1,1, figsize = (5,5))
for ig, gam in enumerate(gammas):
  col = plt.get_cmap("tab10")(ig)
  ax2.semilogy(ni, (cost_spdhg_sino[ig,:] - c_ref)/n, '-.', ms = 5,
               label = f'SINO {gam:.1e}',color = col)
  ax2.semilogy(ni, (cost_spdhg_lm[ig,:] - c_ref)/n, '-v', ms = 5, 
               label = f'LM   {gam:.1e}', color = col)

ax2.grid(ls = ':')
ax2.set_xlabel('iteration')
ax2.set_ylabel('relative cost')

handles, labels = ax2.get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper right', ncol = len(gammas), fontsize = 'small')

fig2.tight_layout()
fig2.show()

# save the figures
fig.savefig(os.path.join('figs',f'{base_str}_precond_{precond}.png'))
fig2.savefig(os.path.join('figs',f'{base_str}_precond_{precond}_metrics.pdf'))
fig2.savefig(os.path.join('figs',f'{base_str}_precond_{precond}_metrics.png'))
