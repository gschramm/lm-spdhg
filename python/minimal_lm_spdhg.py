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
def fwd_lm(x, sens, events):
  return x*sens[events]

#---------------------------------------------------------------------------------------------
def back_lm(y, sens, events): 
  return (y*sens[events]).sum()

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
def SPDHG(sens, b, contam, niter = 100, rho = 0.999, gamma = 1, verbose = False, precond = False):

  # probability that a subset gets chosen (using 2 subsets)
  p_p = 0.5  

  # set step sizes
  if precond:
    S = gamma*rho/sens
    T = p_p*(rho/(gamma*sens)).min()
  else:
    S = [gamma*rho/sens[0], gamma*rho/sens[1]]
    T = p_p*min(rho/(gamma*sens[0]),rho/(gamma*sens[1]))

  x = 0
  y = np.zeros(b.shape)

  z    = back(y, sens)
  zbar = z.copy()

  for i in range(niter):
    ss = np.random.randint(0,2)

    x = np.clip(x - T*zbar, 0, None)
    if verbose: print(i, ss, x)
    
    y_plus = y[ss]+ S[ss]*(fwd(x, sens[ss]) + contam[ss])
    
    # apply the prox for the dual of the poisson logL
    y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S[ss]*b[ss]))
   
    dz = back(y_plus - y[ss], sens[ss])
    
    # update variables
    z     = z + dz
    y[ss] = y_plus.copy()
    zbar  = z + dz/p_p

  return x

#---------------------------------------------------------------------------------------------
def PDHG_LM(sens, events, mu, contam_list, 
            niter = 100, rho = 0.999, gamma = 1, verbose = False, precond = False):
  # set step sizes
  if precond:
    S    = (gamma*rho/fwd_lm(1,sens,events))
    T     = rho/(gamma*back(np.ones(2),sens))
  else:
    norm = np.linalg.norm(sens)
    S    = (gamma*rho/norm)
    T     = rho/(gamma*norm)

  x = 0

  # dual variable for "lists"
  y = np.zeros(events.shape)

  # dual variable for "histograms"
  yh = np.ones(sens.shape)
  yh[events] = 0

  z    = back(yh, sens)
  zbar = z.copy()

  cc = contam_list

  for i in range(niter):
    x = np.clip(x - T*zbar, 0, None)
    if verbose: print(x)
    
    y_plus = y + S*(fwd_lm(x, sens, events) + cc)
    
    # apply the prox for the dual of the poisson logL
    y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S*mu))
    
    dz = back_lm((y_plus - y)/mu, sens, events)
    
    # update variables
    z = z + dz
    y = y_plus.copy()
    zbar = z + dz

  return x



#---------------------------------------------------------------------------------------------

sens   = np.array([1,0.1])
contam = np.array([0.02,0.01])

# simulate data
xtrue = 1*1.2
ytrue = fwd(xtrue, sens) + contam

seeds = np.arange(10)
#seeds = np.array([31])  # intereting case where b = [0,1]

xML       = np.zeros(seeds.shape[0])
xMLEM     = np.zeros(seeds.shape[0])
xSPDHG    = np.zeros(seeds.shape[0])
xPDHG_LM  = np.zeros(seeds.shape[0])

niter = 200

for i, seed in enumerate(seeds):
  np.random.seed(seed)
  
  # generate noise realization (histogram)
  b = np.random.poisson(ytrue)
  
  # generate LM events stream
  events = np.zeros(b.sum(), dtype = np.uint8)
  events[b[0]:] = 1 
  mu            = b[events]

  xML[i]   = ML(sens, b, contam)

  xSPDHG[i]    = SPDHG(sens, b, contam, niter = niter, gamma = 1/xtrue,
                       verbose = False, precond = True)
  xPDHG_LM[i] = PDHG_LM(sens, events, mu, contam[events], niter = niter, gamma = 1/xtrue, 
                        verbose = False, precond = True)

xML2 = np.clip(xML,0,None)

print('SPDHG    max diff vs non-neg ML', np.abs(xSPDHG - xML2).max())
print('PDHG_LM max diff vs non-neg ML', np.abs(xPDHG_LM - xML2).max())

fig, ax = plt.subplots(figsize = (5,5))
ax.plot(xML2,xML2,'-')
ax.plot(xML2,xSPDHG,'.')
ax.plot(xML2,xPDHG_LM,'.')
ax.grid(ls = ':')
fig.tight_layout()
fig.show()
