import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path

fname = Path('data/20211206_paper/TV_3e-2_4e7.npz')
data = np.load(fname)
xm = data['x_early']

ims = {'vmax':1.2*data['img'].max(), 'cmap': plt.cm.Greys}

it = 20
i  = np.argwhere(data['it'] == it)[0][0]

fig = plt.figure(figsize = (16,2.7))
gs = gridspec.GridSpec(2, 7)

ax = fig.add_subplot(gs[:2, :1])
ax.plot(np.arange(1,len(data['cost'])+1), data['cost']/1e8)
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
