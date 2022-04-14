sinofile = '/uz/data/Admin/ngeworkingresearch/mi_patient_data/respmotion/20210930_16918_61360708/raw_idlfiles/rdf.7.1'
sinodescrip = { scanner:            'dmi',      $
                emission:           sinofile,   $
                sensitivity:        sinofile,   $
                attenuation:        sinofile,   $
                scatter_norm:       sinofile,   $
                scatter_ds_scatter: sinofile,   $
                randoms:            sinofile}

listfile = '/uz/data/Admin/ngeworkingresearch/mi_patient_data/respmotion/20210930_16918_61360708/LST/LIST0006.BLF'
listdescrip = {scanner:      'dmi',       $
               acquisition:  'listmode',  $
               listfile:      listfile }

; define a projector using the "real" geometry (no LOR meshing)
projd = nidef_proj(/pet3d_dmi, /raytracer, /list, /binmash, /true_segmentsampling)



; read the corrections sinograms
dummy = niread_data(sinodescrip, projd=projd, /noemis, sens=sens, atten=atten, contam=contam, /verbose)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; generate pseudo attenuation and sens sinograms that we need to caluclate the sensitivity image
;
coords_all = nipet3dcoords(projd.petstruct, /allxtals)
; get the sensitivity and attenuation values for all LORs in LM format
; we need those to calculate the sensivity image
sens_all  = nipet3dcoords(projd.petstruct, /samplesinogram, sinogram=sens)
atten_all = nipet3dcoords(projd.petstruct, /samplesinogram, sinogram=atten)

print, 'sens'
niwrite_hdf5, sens_all, "data/dmi/lm_data.h5", "all_xtals", "sens"
print, 'atten'
niwrite_hdf5, atten_all, "data/dmi/lm_data.h5", "all_xtals", "atten"
print, 'xtals'
niwrite_hdf5, coords_all, "data/dmi/lm_data.h5", "all_xtals", "xtal_ids"

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

print, 'reading LM'
coords = niread_data(listdescrip, projd=projd, indices=indices, /verbose)

if n_elements(sens) gt 1 then sensvalues = reform(sens[indices[0,*],indices[1,*],indices[2,*]])
if n_elements(atten) gt 1 then attenvalues = reform(atten[indices[0,*],indices[1,*],indices[2,*]])
if n_elements(contam) gt 1 then contamvalues = reform(contam[indices[0,*],indices[1,*],indices[2,*],coords[-1,*]])

niwrite_hdf5, sensvalues,   "data/dmi/lm_data.h5", "correction_lists", "sens"
niwrite_hdf5, attenvalues,  "data/dmi/lm_data.h5", "correction_lists", "atten"
niwrite_hdf5, contamvalues, "data/dmi/lm_data.h5", "correction_lists", "contam"

niwrite_hdf5, sinofile, "data/dmi/lm_data.h5", "header", "sinofile"
niwrite_hdf5, listfile, "data/dmi/lm_data.h5", "header", "listfile"

END
