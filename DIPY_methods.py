#!/usr/bin/env python

####################################################################
# Python 2.7 script for executing FA, MD calculations for one case #
####################################################################

#
# Generate streamlines with DIPY
#
# imgfile    - DTI file (.nii.gz)
# maskfile   - Brain mask file (.nii.gz)
# bvals      - b-values file (ASCII)
# bvecs      - tensor directions file (ASCII)
# out_prefix - subject specific prefix
#
def DIPY_nii2streamlines(imgfile, maskfile, bvals, bvecs, output_prefix):
    import numpy as np
    import nibabel as nib
    import os

    from dipy.reconst.dti import TensorModel

    print "nii2streamlines"

    img = nib.load(imgfile)
    bvals = np.genfromtxt(bvals)
    bvecs = np.genfromtxt(bvecs)
    if bvecs.shape[1] != 3:
        bvecs = bvecs.T

    from nipype.utils.filemanip import split_filename
    _, prefix, _  = split_filename(imgfile)
    from dipy.data import gradient_table
    gtab = gradient_table(bvals, bvecs)
    data = img.get_data()
    affine = img.get_affine()
    zooms = img.get_header().get_zooms()[:3]
    new_zooms = (2., 2., 2.)
    data2, affine2 = data, affine
    mask = nib.load(maskfile).get_data().astype(np.bool)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data2, mask)

    from dipy.reconst.dti import fractional_anisotropy
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    fa_img = nib.Nifti1Image(FA, img.get_affine())
    nib.save(fa_img, experiment_dir + '/' + ('%s_tensor_fa.nii.gz' % prefix))
    evecs = tenfit.evecs
    evec_img = nib.Nifti1Image(evecs, img.get_affine())
    nib.save(evec_img, experiment_dir + '/' + ('%s_tensor_evec.nii.gz' % prefix))

    from dipy.data import get_sphere
    sphere = get_sphere('symmetric724')
    from dipy.reconst.dti import quantize_evecs

    peak_indices = quantize_evecs(tenfit.evecs, sphere.vertices)

    from dipy.tracking.eudx import EuDX
    eu = EuDX(FA, peak_indices, odf_vertices = sphere.vertices, a_low=0.2, seeds=10**6, ang_thr=35)
    tensor_streamlines = [streamline for streamline in eu]
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = new_zooms
    hdr['voxel_order'] = 'LPS'
    hdr['dim'] = data2.shape[:3]

    import dipy.tracking.metrics as dmetrics
    tensor_streamlines = ((sl, None, None) for sl in tensor_streamlines if dmetrics.length(sl) > 15)
    ten_sl_fname = experiment_dir + '/' + ('%s_streamline.trk' % prefix)
    nib.trackvis.write(ten_sl_fname, tensor_streamlines, hdr, points_space='voxel')
    return ten_sl_fname