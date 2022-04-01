import pyparallelproj as ppp
import numpy as np
import h5py

from pathlib import Path

import h5py

voxsize = np.array([2.78, 2.78, 2.78], dtype = np.float32)
img     = np.zeros((180,180,71), dtype = np.float32)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# speed of light in mm/ns
speed_of_light = 300.

# time resolution FWHM in ns
time_res_FWHM = 0.385

# sigma TOF in mm
sigma_tof = (speed_of_light/2) * (time_res_FWHM/2.355)

# define the 4 ring DMI geometry
scanner = ppp.RegularPolygonPETScanner(
               R                    = 0.5*(744.1 + 2*8.51),
               ncrystals_per_module = np.array([16,9]),
               crystal_size         = np.array([4.03125,5.31556]),
               nmodules             = np.array([34,4]),
               module_gap_axial     = 2.8)

#scanner.show_crystal_config()


# the TOF bin width in mm is 13*0.01302ns times the speed of light (300mm/ns) divided by two
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 29, tofbin_width = 13*0.01302*speed_of_light/2)

# define the projector

proj = ppp.SinogramProjector(scanner, sino_params, img.shape, 
                             voxsize = voxsize, tof = True, 
                             sigma_tof = sigma_tof, n_sigmas = 3.)



#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------


# read the LM data
pdir = Path('.')
LM_file = pdir / 'LST' / 'LIST0002.BLF'

with h5py.File(LM_file, 'r') as data:
  events = data['MiceList/TofCoinc'][:]

nevents = events.shape[0]

# swap axial and trans-axial crystals IDs
events[:,[0,1]] = events[:,[1,0]]
events[:,[2,3]] = events[:,[3,2]]

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed 

events[:,-1] = -(events[:,-1]//13)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# backproject all LM events
vals    = np.ones(nevents, dtype = np.float32)
bimg    = proj.back_project_lm(vals, events[:nevents,:])

import pymirc.viewer as pv
vi = pv.ThreeAxisViewer(bimg)
