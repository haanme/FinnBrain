#!/usr/bin/env python

import pipelineutils as putils
experiment_dir = '/Users/eija/Documents/FinnBrain'
DTIprep_protocol = '/Users/eija/Documents/FinnBrain/scripts/default.xml'
import nipype.interfaces.camino as cmon

# Convert FSL b-vectors to Camino formatting
def FSL2Scheme(bval_file, bvec_file):
    f2s = cmon.FSL2Scheme()
    f2s.inputs.bval_file = bval_file
    f2s.inputs.bvec_file = bvec_file
    print f2s.cmdline
    r = f2s.run()
    return r.outputs.scheme

# get sigma for restore
def get_sigma(dwi_file, bmask_file):
    import nibabel as nib
    import numpy as np
    from scipy.special import gamma

    img = nib.load(dwi_file)
    dwidata = img.get_data()
    img = nib.load(bmask_file)
    bgdata = img.get_data()
    print "DTI shape " + str(dwidata.shape)
    print "BG shape " + str(bgdata.shape)
    directions = dwidata.shape[3]
    xdim = dwidata.shape[0]
    ydim = dwidata.shape[1]
    zdim = dwidata.shape[2]
    stds = []
    d_i = 0
    values = []
    n = 0
    for z_i in range(zdim):
        for y_i in range(ydim):
            for x_i in range(xdim):
                if bgdata[x_i, y_i, z_i] != 0:
                    values.append(dwidata[x_i, y_i, d_i, z_i])
                    n = n + 1
    std_dir = np.std(values)
    stds.append(std_dir)
    mean_std = np.mean(stds)
    bias = mean_std/(4*n)
    sigma = mean_std + bias
    return sigma

# Convert Nifti to Camino formatting
def Image2Voxel(nifti_file):
    i2v = cmon.Image2Voxel()
    i2v.inputs.in_file = nifti_file
    print i2v.cmdline
    r = i2v.run()
    return r.outputs.voxel_order

# Convert Camino formatting to Nifti
def Voxel2Image(bfloat_file, nii_file_header):

    import os
    import shutil
    from nipype.interfaces.base import CommandLine
    fileName, fileExtension = os.path.splitext(bfloat_file)
    head, root = os.path.split(fileName)
    cmd = CommandLine('cat %s | fa -inputmodel dt | voxel2image -outputroot %s -header %s -components %s' % (bfloat_file, root, nii_file_header, 1))
    print "voxel2image:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s.nii' % (root))

def dt2Image(bfloat_file, nii_file_header):

    import os
    import shutil
    from nipype.interfaces.base import CommandLine
    fileName, fileExtension = os.path.splitext(bfloat_file)
    head, root = os.path.split(fileName)
    cmd = CommandLine('dt2nii -inputfile %s -inputdatatype float -header %s -outputroot camino_' % (bfloat_file, nii_file_header))
    print "dt2nii:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s.nii' % (root))


# RESTORE
def Camimo_ModelFit(scheme_file, bfloat_file, sigma, model):
    
    fit = cmon.ModelFit()
    fit.inputs.model = model
    fit.inputs.scheme_file = scheme_file
    fit.inputs.sigma = sigma
    fit.inputs.in_file = bfloat_file
    head, tail = os.path.split(bfloat_file)
    fit.inputs.out_file = head + '/' + 'data_fit_' + model + '.Bdouble'
    print fit.cmdline
    r = fit.run()
    return r.outputs.fitted_data

#
# Convert b-vectors for usage in fsl
#
# dwifile    - DTI file (.nii.gz)
# bvec       - tensor directions file (ASCII)
# out_prefix - subject specific prefix
#
def correctbvec4fsl(dwifile, bvec, output_prefix):
    import nibabel as nib
    import numpy as np
    from nipype.utils.filemanip import split_filename

    print "correctbvec4fsl"
    aff = nib.load(dwifile).get_affine()[:3, :3]
    for i in range(10):
        #aff = aff.dot(np.linalg.inv(np.eye(3) + 3*aff.T.dot(aff)).dot(3*np.eye(3) + aff.T.dot(aff)))
        aff = 0.5 * (aff + np.linalg.inv(aff.T))
    mat = np.dot(aff, np.array([[1,0,0],[0,1,0],[0,0,-1]])) # DTIPrep output in nifti
    bvecs = np.genfromtxt(bvec)
    if bvecs.shape[1] != 3:
        bvecs = bvecs.T
    bvecs = mat.dot(bvecs.T).T
    outfile = experiment_dir + '/' + output_prefix + '/' + ('%s_forfsl.bvec' % split_filename(bvec)[1])
    np.savetxt(outfile, bvecs, '%.17g %.17g %.17g')
    return outfile

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
    print bvecs.shape

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

from argparse import ArgumentParser
import os
import conversions as conv
import sys
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dicomdir", dest="dicomdir", help="dicom base directory for image data", required=True)
    parser.add_argument("--DTI1", dest="DTI1", help="DTI acquisition part 1 of 3", required=True)
    parser.add_argument("--DTI2", dest="DTI2", help="DTI acquisition part 2 of 3", required=True)
    parser.add_argument("--DTI3", dest="DTI3", help="DTI acquisition part 3 of 3", required=True)
    parser.add_argument("--T1", dest="T1", help="T1 anatomical reference", required=True)
    parser.add_argument("--FieldMap_Mag", dest="FieldMap_Mag", help="Fieldmap Magnitude image", required=True)
    parser.add_argument("--FieldMap_Pha", dest="FieldMap_Pha", help="Fieldmap Phase image", required=True)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    args = parser.parse_args()

    basedir = experiment_dir + '/' + args.subject + '/'
    dwifile = (basedir + 'data.nii.gz')

    RESTORE_sigma = get_sigma(dwifile, (basedir + 'nodif_brain_mask.nii.gz'))
    print RESTORE_sigma

    dwifile_2 = conv.gznii2nii(dwifile)
    print dwifile_2

    bvals = (basedir + 'dwi_LPS_fmt2unwarped.bval')
    bvecs = (basedir + 'dwi_LPS_fmt2unwarped.bvec')
    print bvals
    print bvecs
    corr_bvec = correctbvec4fsl(dwifile_2, bvecs, args.subject)
    tsl_file = DIPY_nii2streamlines(dwifile_2, (basedir + 'nodif_brain_mask.nii.gz'), bvals, corr_bvec, args.subject)
    print tsl_file
    sys.exit(0)

    dwifile_ana_hdr, dwifile_ana_img =conv.nii2analyze(dwifile_2)
    print dwifile_ana_hdr
    print dwifile_ana_img
    print dwifile
    dwifile_cmon = Image2Voxel(dwifile)

    print dwifile_cmon
    scheme_cmon = FSL2Scheme(basedir + 'bvals', basedir + 'bvecs')

    print scheme_cmon
    fitfile_cmon = Camimo_ModelFit(scheme_cmon, dwifile_cmon, RESTORE_sigma, 'restore')
    fitfile = dt2Image(fitfile_cmon, dwifile)
    move_to_results(fitfile, args.subject)
    fitfile_cmon = Camimo_ModelFit(scheme_cmon, dwifile_cmon, RESTORE_sigma, 'pospos dt')
    fitfile = dt2Image(fitfile_cmon, dwifile)
    move_to_results(fitfile, args.subject)


