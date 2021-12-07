import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


x = np.load('data/20211206_paper/TV_7e7_0.03_20it.npy')
ims = {'vmax':15, 'cmap': plt.cm.Greys}

it = [0,19]

fig = plt.figure(figsize = (6,6))
gs = gridspec.GridSpec(4, 2)

ax = fig.add_subplot(gs[:2, 0,])
ax.imshow(x[it[0],:,:,49].T, **ims)
ax.set_ylabel(f'iteration {(it[0]+1):2} / 112 subsets')

ax0 = fig.add_subplot(gs[0, 1])
ax0.imshow(np.flip(x[it[0],122,:,:].T,0), **ims)

ax1 = fig.add_subplot(gs[1, 1])
ax1.imshow(np.flip(x[it[0],:,122,:].T,0), **ims)


ax = fig.add_subplot(gs[2:, 0,])
ax.imshow(x[it[1],:,:,49].T, **ims)
ax.set_ylabel(f'iteration {(it[1]+1):2} / 112 subsets')

ax0 = fig.add_subplot(gs[2, 1])
ax0.imshow(np.flip(x[it[1],122,:,:].T,0), **ims)

ax1 = fig.add_subplot(gs[3, 1])
ax1.imshow(np.flip(x[it[1],:,122,:].T,0), **ims)


for axx in fig.axes:
  axx.set_xticks([])
  axx.set_xticklabels([])
  axx.set_yticks([])
  axx.set_yticklabels([])

fig.tight_layout()
fig.show()
