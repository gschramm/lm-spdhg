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
def fwd_lm(x, sens, events, mu):
  return x*sens[events]/mu

#---------------------------------------------------------------------------------------------
def back_lm(y, sens, events, mu): 
  return (y*sens[events]/mu).sum()

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
def SPDHG(sens, b, contam, niter = 100, rho = 0.999, gamma = 1, verbose = False):

  # probability that a subset gets chosen (using 2 subsets)
  p_p = 0.5  

  S = gamma*rho/sens
  T = p_p*(rho/(gamma*sens)).min()

  x = 0
  y = (b == 0).astype(float)

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
def SPDHG_LM(sens, events, mu, contam_list, 
             niter = 100, rho = 0.999, gamma = 1, verbose = False):

  # probability that a subset gets chosen (using 2 subsets)
  p_p = 0.5  

  S_i = [gamma*rho/fwd_lm(1, sens,events[slice(0,None,2)], mu[slice(0,None,2)]),
         gamma*rho/fwd_lm(1, sens,events[slice(1,None,2)], mu[slice(1,None,2)])]

  if events.shape[0] >= 2:
    T_i = [p_p*rho/(gamma*back_lm(np.ones(events[slice(0,None,2)].shape[0]), 
                                  sens,events[slice(0,None,2)], mu[slice(0,None,2)])),
           p_p*rho/(gamma*back_lm(np.ones(events[slice(1,None,2)].shape[0]), 
                                  sens,events[slice(1,None,2)], mu[slice(1,None,2)]))]

    T = min(T_i)
  else:
    T = p_p*rho/(gamma*back_lm(np.ones(events[slice(0,None,2)].shape[0]), 
                               sens,events[slice(0,None,2)], mu[slice(0,None,2)]))

  x = 0

  # dual variable for "lists"
  y = np.zeros(events.shape)

  # dual variable for "histograms"
  yh = np.ones(sens.shape)
  yh[events] = 0

  z    = back(yh, sens)
  zbar = z.copy()

  for i in range(niter):
    iss = np.random.randint(0,2)
    ss  = slice(iss,None,2)

    x = np.clip(x - T*zbar, 0, None)
    if verbose: print(i, ss, x)

    y_plus = y[ss] + S_i[iss]*(fwd_lm(x, sens, events[ss], mu[ss]) + contam_list[ss]/mu[ss])
    
    # apply the prox for the dual of the poisson logL
    y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[iss]))
    
    dz = back_lm(y_plus - y[ss], sens, events[ss], mu[ss])
    
    # update variables
    z     = z + dz
    y[ss] = y_plus.copy()
    zbar  = z + dz/p_p

  return x

#---------------------------------------------------------------------------------------------

sens   = np.array([1,0.5])
contam = np.array([0.2,0.1])

# simulate data
xtrue = 2
ytrue = fwd(xtrue, sens) + contam

seeds   = np.arange(20)

niter = 500
gamma = 1/xtrue

xML       = np.zeros(seeds.shape[0]) 
xSPDHG    = np.zeros(seeds.shape[0]) 
xSPDHG_LM = np.zeros(seeds.shape[0]) 


for i, seed in enumerate(seeds):
  np.random.seed(seed)
  
  # generate noise realization (histogram)
  b = np.random.poisson(ytrue)

  # we need >= 2 counts in order to have 2 subsets in LM
  # analystic ML solution
  xML[i] = ML(sens, b, contam)
  # histogram SPDHG
  xSPDHG[i] = SPDHG(sens, b, contam, niter = niter, gamma = gamma, verbose = False)
 
  if b.sum() > 0:
    # generate LM events stream
    events = np.zeros(b.sum(), dtype = np.uint8)
    events[b[0]:] = 1 

    # shuffle the LM events
    tmp = np.arange(events.shape[0])
    np.random.shuffle(tmp)
    events = events[tmp]

    # counts for each LOR in LM stream
    mu = b[events]

    xSPDHG_LM[i] = SPDHG_LM(sens, events, mu, contam[events], niter = niter, gamma = gamma, 
                            verbose = False)
  else:
    xSPDHG_LM[i] = 0
  

xML2 = np.clip(xML,0,None)

print('SPDHG    max diff vs non-neg ML', np.abs(xSPDHG - xML2).max())
print('SPDHG_LM max diff vs non-neg ML', np.abs(xSPDHG_LM - xML2).max())

fig, ax = plt.subplots(figsize = (5,5))
ax.plot([xML2.min(),xML2.max()],[xML2.min(),xML2.max()],'k-',lw=0.5,label='identity')
ax.plot(xML2,xSPDHG, 'x', label = 'SPDHG', ms = 8)
ax.plot(xML2,xSPDHG_LM, '.', label = 'LM SPDHG')
ax.grid(ls = ':')
ax.legend()
ax.set_xlabel('analytic (non-neg.) ML solution')
ax.set_ylabel('SPDHG solution')
fig.tight_layout()
fig.show()
