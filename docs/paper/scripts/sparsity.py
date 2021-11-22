import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.rcParams['font.size'] = 16

with h5py.File('../data/rdf.4.1', 'r') as data:
  view = data['/emission/view11'][:]


cm = ListedColormap(np.array([[1.0, 1.0, 1.0, 1.],
                              [109/225, 171/255, 205/255, 1.],
                              [1.0, 0.0, 0.0, 1.],
                              [1.0, 1.0, 0.0, 1.],
                              [ 90/225, 196/255,  90/255, 1.],
                              [1.0, 0.0, 1.0, 1.],
                              [0.0, 1.0, 1.0, 1.],
                              [0.0, 0.0, 0.0, 1.]]))


fig, ax = plt.subplots(3,1, figsize = (15,3*2.9))
im0 = ax[0].imshow(view.sum(0),  vmax = 8, cmap = cm)
plt.colorbar(im0, ax = ax[0], fraction = 0.01, shrink = 1.0, aspect = 25)
ax[0].set_title(f'sum over all TOF bins- sparsity {(view.sum(0) == 0).sum() / view.sum(0).size:.3f}')

i = 14
im1 = ax[1].imshow(view[i,...], vmax = 8, cmap = cm)
plt.colorbar(im1, ax = ax[1], fraction = 0.01, shrink = 1.0, aspect = 25)
ax[1].set_title(f'TOF bin {i+1}/29 - sparsity {(view[i,...] == 0).sum() / view[i,...].size:.3f}')

i = 21
im2 = ax[2].imshow(view[20,...], vmax = 8, cmap = cm)
plt.colorbar(im2, ax = ax[2], fraction = 0.01, shrink = 1.0, aspect = 25)
ax[2].set_title(f'TOF bin {i+1}/29 - sparsity {(view[i,...] == 0).sum() / view[i,...].size:.3f}')


for axx in ax.ravel():
  axx.set_xticks([])
  axx.set_yticks([])

  axx.set_ylabel('axial direction', fontsize = 'large')

ax[-1].set_xlabel('radial direction', fontsize = 'large')

fig.tight_layout()
fig.savefig('../figs/sparsity.png')
fig.show()

