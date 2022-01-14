import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path

fname = Path('data/20220103_paper/counts_4e7_beta_0.03_rho_1.npz')
data = np.load(fname)
xm   = data['x_early']
cost = data['cost_lm'] 

xm2   = data['x_early2']
cost2 = data['cost_lm2'] 

ims = {'vmax':1.2*data['img'].max(), 'cmap': plt.cm.Greys}

it = 20
i  = np.argwhere(data['it'] == it)[0][0]

#-------------------------------------------------------------------------------

fig = plt.figure(figsize = (0.8*16,0.8*2.7))
gs = gridspec.GridSpec(2, 7)

ax = fig.add_subplot(gs[:2, :1])
ax.plot(np.arange(1,len(cost)+1), cost/1e8)
ax.set_xlabel('iteration')
ax.set_ylabel('cost / 1e8')
ax.grid(ls = ':')
ax.set_xlim(0,50)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 


ax = fig.add_subplot(gs[:2, 1])
ax.imshow(xm[i,:,:,49].T, **ims)
ax.set_title(f'iteration {it:3} / 224 subsets', fontsize = 'medium')
ax0 = fig.add_subplot(gs[0, 2])
ax0.imshow(np.flip(xm[i,122,:,:].T,0), **ims)
ax1 = fig.add_subplot(gs[1, 2])
ax1.imshow(np.flip(xm[i,:,122,:].T,0), **ims)

ax = fig.add_subplot(gs[:3, 3])
ax.imshow(xm[-1,:,:,49].T, **ims)
ax.set_title(f'iteration {data["it"][-1]} / 224 subsets', fontsize = 'medium')
ax0 = fig.add_subplot(gs[0, 4])
ax0.imshow(np.flip(xm[-1,122,:,:].T,0), **ims)
ax1 = fig.add_subplot(gs[1, 4])
ax1.imshow(np.flip(xm[-1,:,122,:].T,0), **ims)

ax = fig.add_subplot(gs[:2, 5])
ax.imshow(data['img'][:,:,49].T, **ims)
ax.set_title(f'ground truth', fontsize = 'medium')
ax0 = fig.add_subplot(gs[0, 6])
ax0.imshow(np.flip(data['img'][122,:,:].T,0), **ims)
ax1 = fig.add_subplot(gs[1, 6])
ax1.imshow(np.flip(data['img'][:,122,:].T,0), **ims)

for axx in fig.axes[1:]:
  axx.set_xticks([])
  axx.set_xticklabels([])
  axx.set_yticks([])
  axx.set_yticklabels([])

fig.tight_layout()
fig.savefig(fname.with_suffix('.png'))
fig.show()

#-------------------------------------------------------------------------------

vm   = 0.05*xm2[-1,...,49].max()
ims2 = {'vmax':vm, 'vmin':-vm, 'cmap': plt.cm.bwr}

fig2, ax2 = plt.subplots(4, xm.shape[0], figsize = (1.2*xm.shape[0],1.2*4))

for i in range(xm.shape[0]):
  ax2[0,i].imshow(xm[i,...,49].T, **ims)
  ax2[1,i].imshow(xm2[i,...,49].T, **ims)
  ax2[2,i].imshow(xm[i,...,49].T - xm2[-1,...,49].T, **ims2)
  ax2[3,i].imshow(xm2[i,...,49].T - xm2[-1,...,49].T, **ims2)

  ax2[0,i].set_axis_off()
  ax2[1,i].set_axis_off()
  ax2[2,i].set_axis_off()
  ax2[3,i].set_axis_off()

  ax2[0,i].set_title(f'iteration {data["it"][i]}', fontsize = 'small')

fig2.tight_layout() 
fig2.savefig(str(fname.with_suffix('')) + '_convergence.png')
fig2.show()

#-------------------------------------------------------------------------------

fig3, ax3 = plt.subplots(1,2, figsize = (8,3), sharex = True)
ax3[0].plot(np.arange(1,len(cost)+1), cost/1e8, label = r'$\rho = 0.999, \gamma = 3 / \| x^0 \|_\infty$')
ax3[0].plot(np.arange(1,len(cost2)+1), cost2/1e8, label = r'$\rho = 8, \gamma = 30 / \| x^0 \|_\infty$')
ax3[1].plot(np.arange(1,len(cost)+1), cost/1e8)
ax3[1].plot(np.arange(1,len(cost2)+1), cost2/1e8)
ax3[1].set_ylim(cost2.min()/1e8, data['c0']/1e8)
ax3[0].legend()
ax3[0].set_ylabel('cost / 1e8')
ax3[1].set_ylabel('(zoom)   cost / 1e8')

for axx in ax3.ravel():
  axx.set_xlabel('iteration')
  axx.grid(ls = ':')

fig3.tight_layout() 
fig3.savefig(str(fname.with_suffix('')) + '_convergence_metrics.png')
fig3.show()
