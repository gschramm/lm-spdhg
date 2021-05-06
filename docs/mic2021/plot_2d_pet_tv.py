import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

base_str = 'brain2d_counts_1.0E+06_beta_2.0E-03_niter_10000_100_nsub_56'
subdir   = os.path.join('..','..','python','data','20210506')
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

fs = (0.8*3.2*(len(gammas)+1),4*3.2)
fig = plt.figure(figsize = fs)
grid  = ImageGrid(fig, (0.05,0,0.95,1),  # similar to subplot(111)
                 nrows_ncols=(len(gammas)+1, 4),  # creates 2x2 grid of axes
                 axes_pad=0.15,  # pad between axes in inch.
                 cbar_mode = 'edge',
                 cbar_location = 'left'
                 )

for i,gam in enumerate(gammas):
  i0 = grid[0*4+i].imshow(x_sino[i,...].squeeze()[sl], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
  i1 = grid[1*4+i].imshow(x_lm[i,...].squeeze()[sl],   vmin = 0, vmax = vmax, cmap = plt.cm.Greys)

  i2 = grid[2*4+i].imshow(x_sino[i,...].squeeze()[sl] - ref_recon.squeeze()[sl], 
                     vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)
  i3 = grid[3*4+i].imshow(x_lm[i,...].squeeze()[sl] - ref_recon.squeeze()[sl],   
                 vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = plt.cm.bwr)
  grid[0*4+i].set_title(f'SPDHG {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  grid[1*4+i].set_title(f'LM-SPDHG {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  grid[2*4+i].set_title(f'diff. SPDHG {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))
  grid[3*4+i].set_title(f'diff. LM-SPDHG {gam:.1e}', fontsize = 'medium', color = plt.get_cmap("tab10")(i))

grid.cbar_axes[0].colorbar(i0)
grid.cbar_axes[1].colorbar(i1)
grid.cbar_axes[2].colorbar(i2)
grid.cbar_axes[3].colorbar(i3)

i01 = grid[0*4+3].imshow(ref_recon.squeeze()[sl], vmin = 0, vmax = vmax, cmap = plt.cm.Greys)
i11 = grid[1*4+3].imshow(img.squeeze()[sl],       vmin = 0, vmax = vmax, cmap = plt.cm.Greys)

grid[0*4+3].set_title(f'reference PDHG', fontsize = 'medium')
grid[1*4+3].set_title(f'ground truth', fontsize = 'medium')

for axx in grid:
  axx.set_axis_off()

fig.show()

c_ref = cost_ref.min()
n     = c_0 - c_ref
ni    = np.arange(niter) + 1

fig2, ax2 = plt.subplots(1,1, figsize = (5,2.5))
for ig, gam in enumerate(gammas):
  col = plt.get_cmap("tab10")(ig)
  ax2.semilogy(ni[::2], ((cost_spdhg_sino[ig,:] - c_ref)/n)[::2], '-', ms = 5,
               label = f'SINO {gam:.1e}',color = col)
  ax2.semilogy(ni[::2], ((cost_spdhg_lm[ig,:] - c_ref)/n)[::2], '-o', ms = 2.5, 
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
