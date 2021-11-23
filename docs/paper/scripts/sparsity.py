import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.rcParams['font.size'] = 24

plt.rcParams['font.sans-serif'] = ['Arial']

with h5py.File('../data/rdf.4.1', 'r') as data:
  view = data['/emission/view11'][:]

vmax = int(view.sum(0).max() + 1)

ca = np.array([[1         , 1         , 1         , 1.        ],
               [0.12156863, 0.46666667, 0.70588235, 1.        ],
               [1.        , 0.49803922, 0.05490196, 1.        ],
               [0.17254902, 0.62745098, 0.17254902, 1.        ],
               [0.83921569, 0.15294118, 0.15686275, 1.        ],
               [0.58039216, 0.40392157, 0.74117647, 1.        ],
               [0.54901961, 0.3372549 , 0.29411765, 1.        ],
               [0.        , 0.        , 0.        , 1.        ],
               [0.7372549 , 0.74117647, 0.13333333, 1.        ],
               [0.49803922, 0.49803922, 0.49803922, 1.        ],
               [0.09019608, 0.74509804, 0.81176471, 1.        ]])[:vmax,:]

cm = ListedColormap(ca)


fig, ax = plt.subplots(3,1, figsize = (15,3*2.9))
im0 = ax[0].imshow(view.sum(0),  vmax = vmax, cmap = cm)
#ax[0].set_title(f'sum over all TOF bins- sparsity {(view.sum(0) == 0).sum() / view.sum(0).size:.3f}')
ax[0].set_ylabel(f'summed TOF bins')

i = 14
im1 = ax[1].imshow(view[i,...], vmax = vmax, cmap = cm)
#ax[1].set_title(f'TOF bin {i+1}/29 - sparsity {(view[i,...] == 0).sum() / view[i,...].size:.3f}')
ax[1].set_ylabel(f'TOF bin {i+1}/29')

i = 21
im2 = ax[2].imshow(view[20,...], vmax = vmax, cmap = cm)
#ax[2].set_title(f'TOF bin {i+1}/29 - sparsity {(view[i,...] == 0).sum() / view[i,...].size:.3f}')
ax[2].set_ylabel(f'TOF bin {i+1}/29')


for axx in ax.ravel():
  axx.set_xticks([])
  axx.set_yticks([])

#ax[1].set_ylabel('plane', fontsize = 'large')
#ax[-1].set_xlabel('radial bin', fontsize = 'large')

fig.tight_layout()
fig.colorbar(im2, ax = ax.ravel().tolist(), fraction = 0.01, aspect = 55, pad = 0.01)

fig.savefig('../figs/sparsity.png')
fig.show()

