;TODO create dummy event list that contains every geometrical LOR exactly once
;     and get corresponding sens/attn sino indices + values for sens image calc.

projd = nidef_proj(/pet3d_dmi, /raytracer, /list, /binmash, /true_segmentsampling)


listfile = '/uz/data/Admin/ngeworkingresearch/mi_patient_data/respmotion/20210930_16918_61360708/LST/LIST0006.BLF'
listdescrip = {scanner:      'dmi',       $
               acquisition:  'listmode',  $
               listfile:      listfile }


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

events = niread_hdf5(listdescrip.listfile, 'MiceList', 'TofCoinc', /list)
events = events[[1,0,3,2,4],*]
               
tofbin_orig = events[4,*] ; could get flipped
tofbin = tofbin_orig
dummy = nipet3dcoords(projd.petstruct, /verbose, $
                      idx1 = events[0,*], ring1 = events[1,*], $
                      idx2 = events[2,*], ring2 = events[3,*], $
                      x1, y1, z1, x2, y2, z2, $
                      radial = radial, angle = angle, plane = plane, bin = tofbin)

; mesh and shift the original tofbins 
tofbin_orig = (tofbin_orig / 13) + 14

tofbin      = (projd.nrbins - 1) - tofbin      ; difference in definitions!
tofbin_orig = (projd.nrbins - 1) - tofbin_orig ; difference in definitions!

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

sinofile = '/uz/data/Admin/ngeworkingresearch/mi_patient_data/respmotion/20210930_16918_61360708/raw_idlfiles/rdf.7.1'
sinodescrip = { scanner:            'dmi',      $
                emission:           sinofile,   $
                sensitivity:        sinofile,   $
                attenuation:        sinofile,   $
                scatter_norm:       sinofile,   $
                scatter_ds_scatter: sinofile,   $
                randoms:            sinofile}

coords = niread_data(listdescrip, projd=projd, indices=indices, /verbose)

dummy = niread_data(sinodescrip, projd=projd, /noemis, sens=sens, atten=atten, contam=contam, /verbose)
;if n_elements(sens) gt 1 then sensvalues = reform(sens[indices[0,*],indices[1,*],indices[2,*]])
;if n_elements(atten) gt 1 then attenvalues = reform(atten[indices[0,*],indices[1,*],indices[2,*]])
;if n_elements(contam) gt 1 then contamvalues = reform(contam[indices[0,*],indices[1,*],indices[2,*],coords[-1,*]])

END
