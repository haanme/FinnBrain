#!/usr/bin/env python

####################################################################
# Python 2.7 script for executing FA, MD calculations for one case #
####################################################################

# Directory where result data are located
experiment_dir = ''
# Protocol that is applied in DTIprep
DTIprep_protocol = ''

#
# Moves file to results folder, overwriting the existing file
#
# filename   - file to be moved
# out_prefix - subject specific prefix
#
def move_to_results(filename, out_prefix):
    import os
    import shutil
    outfile = experiment_dir + '/' + out_prefix + '/' + os.path.basename(filename)
    if os.path.isfile(outfile):
        os.remove(outfile)
    shutil.move(filename,outfile)
    return outfile

#
# Gunzips file to results folder, overwriting the existing file
#
# filename   - file to be moved (.nii.gz)
# out_prefix - subject specific prefix
#
def gunzip_to(filename, out_prefix, destination):
    import os
    import shutil
    from nipype.interfaces.base import CommandLine
    
    cmd = CommandLine('gunzip -f %s' % (filename))
    print "gunzip NII.GZ:" + cmd.cmd
    cmd.run()
    
    basename = os.path.basename(filename[:len(filename)-3])
    outfile = destination + '/' + basename
    if os.path.isfile(outfile):
        os.remove(outfile)
    shutil.move(filename[:len(filename)-3],outfile)
    return outfile

#
# Convert dicom 2 nrrd
#
# dicomdir   - DICOM directory for input
# out_prefix - subject specific prefix
#
def dicom2nrrd(dicomdir, out_prefix):
    import os
    from nipype.interfaces.base import CommandLine
    cmd = CommandLine('DWIConvert --inputDicomDirectory %s --outputVolume %s/%s.nrrd' % (dicomdir, experiment_dir,out_prefix))
    print "DICOM->NRRD:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s/%s.nrrd' % (experiment_dir,out_prefix))

#
# Executes DTIPrep
#
# in_file    - DTI file for QC (.nrrd)
# out_prefix - subject specific prefix
#
def dtiprep(in_file, output_prefix):
    from glob import glob
    import os
    from nipype.interfaces.base import CommandLine
    cmd = CommandLine('DTIPrepExec -c -d -f %s -n %s/%s_notes.txt -p %s -w %s/%s.nrrd' % ((experiment_dir + '/' + output_prefix),(experiment_dir + '/' + output_prefix),output_prefix,DTIprep_protocol,experiment_dir,output_prefix))
    print "DTIPREP:" + cmd.cmd
    cmd.run()
    qcfile = experiment_dir + '/' + output_prefix + '/' + output_prefix + '_QCed.nrrd'
    xmlfile = experiment_dir + '/' + output_prefix + '/' + output_prefix + '_XMLQCResult.xml'
    sumfile = experiment_dir + '/' + output_prefix + '/' + output_prefix + '_QCReport.txt'
    return qcfile, xmlfile, sumfile

#
# Converts NRRD to FSL Nifti format (Nifti that is gzipped)
#
# in_file    - NRRD file to convert
# out_prefix - subject specific prefix
#
def nrrd2nii(in_file, output_prefix):
    from os.path import abspath as opap
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    _, name, _ = split_filename(in_file)
    out_vol = experiment_dir + '/' + output_prefix + '/' + ('%s.nii.gz' % name)
    out_bval = experiment_dir + '/' + output_prefix + '/' + ('%s.bval' % name)
    out_bvec = experiment_dir + '/' + output_prefix + '/' + ('%s.bvec' % name)
    
    cmd = CommandLine(('DWIConvert --inputVolume %s --outputVolume %s --outputBValues %s'
                       ' --outputBVectors %s --conversionMode NrrdToFSL') % (in_file, out_vol,
                                                                             out_bval, out_bvec))

    print "NRRD->NIFTI:" + cmd.cmd
    cmd.run()
    return opap(out_vol), opap(out_bval), opap(out_bvec)

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
# BET Brain extraction
#
# dwifile    - DTI file (.nii.gz)
# out_prefix - subject specific prefix
#
def runbet(dwifile, output_prefix):
    from nipype.interfaces.fsl import BET
    import shutil
    import os
    import os.path
    
    print "runbet"
    res = BET(in_file=dwifile, frac=0.5, mask=True).run()
    move_to_results(res.outputs.out_file, output_prefix)
    out_mask = move_to_results(res.outputs.mask_file, output_prefix)
    return out_mask

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

#
# Tensor fitting with FSL's dtfit
#
# imgfile    - DTI file (.nii.gz)
# maskfile   - brain mask (.nii.gz)
# bvals      - b-values file (ASCII)
# bvecs      - b-vector directions file (ASCII)
# out_prefix - subject specific prefix
#
# FA_out     - Fractional Anisotrophy map
# MD_out     - Mean Diffusivity map
#
def FSL_dtifit(imgfile, maskfile, bvals, bvecs, output_prefix):
    from nipype.interfaces import fsl
    dti = fsl.DTIFit()
    dti.inputs.dwi = imgfile
    dti.inputs.bvecs = bvecs
    dti.inputs.bvals = bvals
    dti.inputs.base_name = output_prefix+'_FSLdtifit'
    dti.inputs.mask = maskfile
    print "FSL dtifit:"+dti.cmdline
    res = dti.run()
    
    FA_out = move_to_results(res.outputs.FA, output_prefix)
    move_to_results(res.outputs.L1, output_prefix)
    move_to_results(res.outputs.L2, output_prefix)
    move_to_results(res.outputs.L3, output_prefix)
    MD_out = move_to_results(res.outputs.MD, output_prefix)
    move_to_results(res.outputs.MO, output_prefix)
    move_to_results(res.outputs.S0, output_prefix)
    move_to_results(res.outputs.V1, output_prefix)
    move_to_results(res.outputs.V2, output_prefix)
    move_to_results(res.outputs.V3, output_prefix)
    #    move_to_results(res.outputs.tensor, output_prefix)
    
    return FA_out, MD_out

#
# Masking with FSL's AppllyMask
#
# filename   - file that is masked (.nii.gz)
# maskfile   - mask file (.nii.gz)
# out_prefix - subject specific prefix
#
def FSL_ApplyMask(filename, maskfile, output_prefix):
    from nipype.interfaces.fsl import maths
    applym = maths.ApplyMask()
    applym.inputs.in_file = filename
    applym.inputs.mask_file = maskfile
    print "ApplyMask:"+applym.cmdline
    res = applym.run()
    outfile = move_to_results(res.outputs.out_file, output_prefix)
    return outfile

#
# Make rigid co-registration with FSL's FLIRT
#
# input_file     - file that is co-registered (.nii.gz)
# reference_file - file where input is moved (.nii.gz)
# out_prefix     - subject specific prefix
#
def FSL_FLIRT_estim(input_file, reference_file, output_prefix):
    from nipype.interfaces import fsl
    import os

    flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt.inputs.in_file =input_file
    flt.inputs.reference = reference_file
    print "FLIRT [" + os.path.basename(input_file) + "->" + os.path.basename(reference_file) + "]:" + flt.cmdline
    res = flt.run()
    outfile = move_to_results(res.outputs.out_file, output_prefix)
    outmatrixfile = move_to_results(res.outputs.out_matrix_file, output_prefix)
    return outfile, outmatrixfile

#
# Apply rigid co-registration with FSL's FLIRT
#
# input_file     - file that is co-registered (.nii.gz)
# reference_file - file where input is moved (.nii.gz)
# out_prefix     - subject specific prefix
#
def FSL_FLIRT_write(input_file, reference_file, matrix_file, output_prefix):
    from nipype.interfaces import fsl
    import os
    
    flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt.inputs.in_file =input_file
    flt.inputs.reference = reference_file
    flt.inputs.in_matrix_file = matrix_file
    flt.inputs.apply_xfm = True
    print "FLIRT [" + os.path.basename(input_file) + "->" + os.path.basename(reference_file) + "]:" + flt.cmdline
    res = flt.run()
    outfile = move_to_results(res.outputs.out_file, output_prefix)
    return outfile

###############
# Main script #
###############
from argparse import ArgumentParser
import glob
import sys
import os
import conversions as conv
if __name__ == "__main__":
    # Parse input arguments into args structure
    parser = ArgumentParser()
    parser.add_argument("--T1", dest="T1", help="T1 subdirectory under subject", required=False)
    parser.add_argument("--DTI", dest="DTI", help="DTI subdirectory under subject", required=False)
    args = parser.parse_args()
    if not args.T1:
        args.T1 = 'MR_T1W'
        print "Using default T1 subfolder:"+args.T1
    if not args.DTI:
        args.DTI = 'MR_DTI_32'
        print "Using default DTI subfolder:"+args.DTI

    # Directory where result data are located
    experiment_dir = '..' + os.sep + 'pipelinedata'
    # Protocol that is applied in DTIprep
    DTIprep_protocol = './default_all.xml'

    datadir = '..' + os.sep + 'data'
    subject_dirs = glob.glob(datadir + os.sep + '*')
    for subject_dir in subject_dirs:
        splitted = subject_dir.split(os.sep)
        subject = splitted[-1]
        DTIdir = subject_dir + os.sep + args.DTI
        T1dir = subject_dir + os.sep + args.T1
        print subject
        print DTIdir
        print T1dir

        # Convert DICOM->NRRD
        out_file = dicom2nrrd(DTIdir, subject)
        # DTIprep QC-tool
        qcfile, _, _ = dtiprep(out_file, subject)
        # Convert NRRD->NII
        dwifile, bval_file, bvec_file = nrrd2nii(qcfile, subject)
        dwifile = experiment_dir + os.sep + subject + os.sep + subject + '_QCed.nii.gz'
        bval_file = experiment_dir + os.sep + subject + os.sep + subject + '_QCed.bval'
        bvec_file = experiment_dir + os.sep + subject + os.sep + subject + '_QCed.bvec'
        mask_file = runbet(dwifile, subject)

        T1_file, outfile_bval, outfile_bvec = conv.dicom2nii(T1dir, subject, 'T1', experiment_dir)
        print T1_file
        T1_file_mask = runbet(T1_file, subject)
        T1_at_FA_file, T1_to_FA_matrix = FSL_FLIRT_estim(T1_file, dwifile, subject)
        T1_mask_file = FSL_FLIRT_write(T1_file_mask, dwifile, T1_to_FA_matrix, subject)

        # Analyze tensors with FSL's dtifit
        FAfile, MDfile = FSL_dtifit(dwifile, mask_file, bval_file, bvec_file, subject)
        gunzip_to(FAfile, subject, experiment_dir)
        gunzip_to(MDfile, subject, experiment_dir)

