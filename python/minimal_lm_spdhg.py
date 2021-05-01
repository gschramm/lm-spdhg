# minimal example for LM SPDHG for poisson distributed data with trivial fwd model

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

#---------------------------------------------------------------------------------------------
def fwd(x, sens):
  return x*sens

#---------------------------------------------------------------------------------------------
def back(y, sens): 
  return (y*sens).sum()

#---------------------------------------------------------------------------------------------
def cost(x, sens, b, contam):
  x_fwd = fwd(x, sens) + contam

  return x_fwd[0] - b[0]*np.log(x_fwd[0]) + x_fwd[1] - b[1]*np.log(x_fwd[1])

#---------------------------------------------------------------------------------------------
def ML(sens, b, contam):
  """ analystic expression for ML solution
      sens   ... system matrix
      b      ... measurement
      contam ... additive contaminations
  """
  A = (sens[0] + sens[1])*sens[0]*sens[1]
  B = (sens[0] + sens[1])*(sens[0]*contam[1] + sens[1]*contam[0]) - sens[0]*sens[1]*(b[0] + b[1])
  C = (sens[0] + sens[1])*contam[0]*contam[1] - sens[0]*b[0]*contam[1] - sens[1]*b[1]*contam[0]
  
  p = B/A
  q = C/A
  
  return (-p/2) + np.sqrt(((p**2)/4) - q)

#---------------------------------------------------------------------------------------------
def MLEM(x, sens, b, contam, niter = 100):
  back_ones = back(np.ones(2), sens)

  for i in range(niter):
    x_fwd = fwd(x,sens) + contam
    x    *= (back(b/x_fwd, sens)/ back_ones)

  return x

#---------------------------------------------------------------------------------------------
def PDHG(sens, b, contam, niter = 100, rho = 0.999, gamma = 1, verbose = False, precond = True):
  # set step sizes
  if precond:
    S    = (gamma*rho/fwd(1,sens))
    T     = rho/(gamma*back(np.ones(2),sens))
  else:
    norm = np.linalg.norm(sens)
    S    = (gamma*rho/norm)
    T     = rho/(gamma*norm)

  x = 0
  y = np.zeros(b.shape)

  z    = back(y, sens)
  zbar = z.copy()

  for i in range(niter):
    x = np.clip(x - T*zbar, 0, None)
    if verbose: print(x)
    
    y_plus = y + S*(fwd(x,sens) + contam)
    
    # apply the prox for the dual of the poisson logL
    y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S*b))
    
    dz = back(y_plus - y, sens)
    
    # update variables
    z = z + dz
    y = y_plus.copy()
    zbar = z + dz

  return x


#---------------------------------------------------------------------------------------------

sens   = np.array([1,0.1])
contam = np.array([0.01,0.01])

# simulate data
xtrue = 1*1.2
ytrue = fwd(xtrue, sens) + contam

seeds = np.arange(1)
#seeds = np.array([31])  # intereting case where b = [0,1]

xML   = np.zeros(seeds.shape[0])
xMLEM = np.zeros(seeds.shape[0])
xPDHG = np.zeros(seeds.shape[0])

for i, seed in enumerate(seeds):
  np.random.seed(seed)
  
  # generate noise realization
  b = np.random.poisson(ytrue)
  
  xML[i]   = ML(sens, b, contam)
  xPDHG[i] = PDHG(sens, b, contam, niter = 100, verbose = False, precond = True)

xML2 = np.clip(xML,0,None)

print('PDHG max diff vs unconstrained ML', np.abs(xPDHG - xML).max())
print('PDHG max diff vs   non-neg     ML', np.abs(xPDHG - xML2).max())

fig, ax = plt.subplots(figsize = (5,5))
ax.plot(xML2,xML2,'-')
ax.plot(xML2,xPDHG,'.')
ax.grid(ls = ':')
fig.tight_layout()
fig.show()
