#!/usr/bin/env python

experiment_dir = '/Users/eija/Documents/FinnBrain/pipelinedata'
DTIprep_protocol = '/Users/eija/Documents/FinnBrain/scripts/default.xml'


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
# Copies file to results folder, overwriting the existing file
#
# filename     - file to be moved
# new_filename - new filename
# out_prefix   - subject specific prefix
#
def copy_to_results(filename, new_filename, out_prefix):
    import os
    import shutil
    outfile = experiment_dir + '/' + out_prefix + '/' + new_filename
    if os.path.isfile(outfile):
        os.remove(outfile)
    print "copying [" + filename + "]->[" + outfile + "]"
    shutil.copy(filename,outfile)
    return outfile

#
# Split filename into (<root>/<basename>.<extension>)
#
# filename - filename that is splitted
#
def split_nii_gz(filename):
    import os
    # split path and filename
    root, basename = os.path.split(filename)
    # split extensions until none is found, catenating extensions
    basename, ext_new = os.path.splitext(basename)
    ext = ext_new
    while len(ext_new) > 0:
        basename, ext_new = os.path.splitext(basename)
        ext = ext_new + ext
    return root, basename, ext

#
# Renames file basename in its location
#
# filename     - filename that is renamed
# basename_new - new basename
#
def rename_basename_to(filename, basename_new):
    import os
    import shutil
    root, basename, ext = split_nii_gz(filename)
    filename_new = os.path.join(root, (basename_new + ext))
    if os.path.isfile(filename_new):
        os.remove(filename_new)
    shutil.move(filename,filename_new)
    return filename_new

#
# Aligns headers of two Nifti files
#
# filename   - file that wil have its heaer hanged to reference
# reference  - reference file
# out_prefix - subject specific prefix
#
def align_nii_headers(filename, reference, output_prefix):
    import nibabel as nib

    nii_file = nib.load(filename)
    nii_ref = nib.load(reference)
    hdr_file = nii_file.get_header()
    hdr_ref = nii_ref.get_header()

    import nipype.interfaces.spm.utils as spmu
    r2ref = spmu.ResliceToReference()
    r2ref.inputs.in_files = filename
    r2ref.inputs.target = reference
    res = r2ref.run()

    nii_res = nib.load(('w' + output_prefix + '_FM1.nii'))
    hdr_res = nii_res.get_header()

    # dimension 1 marsk number of significant dimensions
    print '----------------------'
    print hdr_file['dim']
    print '----------------------'
    print hdr_ref['dim']
    print '----------------------'
    print hdr_res['dim']
    print '----------------------'
    # spacing starts from 2nd index
    print '----------------------'
    print hdr_file['pixdim']
    print '----------------------'
    print hdr_ref['pixdim']
    print '----------------------'
    print hdr_res['pixdim']
    print '----------------------'
    # quaterniuon representation
    print hdr_file['quatern_b']
    print hdr_file['quatern_c']
    print hdr_file['quatern_d']
    print '----------------------'
    print hdr_ref['quatern_b']
    print hdr_ref['quatern_c']
    print hdr_ref['quatern_d']
    print '----------------------'
    print hdr_res['quatern_b']
    print hdr_res['quatern_c']
    print hdr_res['quatern_d']
    print '----------------------'

    print hdr_file['qoffset_x']
    print hdr_file['qoffset_y']
    print hdr_file['qoffset_z']
    print '----------------------'
    print hdr_ref['qoffset_x']
    print hdr_ref['qoffset_y']
    print hdr_ref['qoffset_z']
    print '----------------------'
    print hdr_res['qoffset_x']
    print hdr_res['qoffset_y']
    print hdr_res['qoffset_z']
    print '----------------------'
    # 3x4 transformation matrix (rows 1-3 of 4x4 matrix)
    print hdr_file['srow_x']
    print hdr_file['srow_y']
    print hdr_file['srow_z']
    print '----------------------'
    print hdr_ref['srow_x']
    print hdr_ref['srow_y']
    print hdr_ref['srow_z']
    print '----------------------'
    print hdr_res['srow_x']
    print hdr_res['srow_y']
    print hdr_res['srow_z']
    print '----------------------'

#
# Merge DTI files
#
# filelist - list of files (list of .nrrd files)
# out_prefix - subject specific prefix
#
def dtimerge(filelist, output_prefix):
    import numpy as np
    import sys
    import nrrd
    import StringIO
    import dtimerge

    outputname = (experiment_dir + '/' + output_prefix + '/' + output_prefix + 'merged.nrrd')
    print "DTIMERGE " + ('[%s]' % ', '.join(map(str, filelist)))
    dtimerge.dtimerge(filelist,outputname)
    return outputname

#
# Run DTIPrep
#
# in_file    - input DTI file (.nrrd)
# out_prefix - subject specific prefix
#
def dtiprep(in_file, output_prefix):
    from glob import glob
    import os
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    cmd = CommandLine('DTIPrepExec -c -d -f %s -n %s/%s_notes.txt -p %s -w %s' % ((experiment_dir + '/' + output_prefix),(experiment_dir + '/' + output_prefix),output_prefix,DTIprep_protocol,in_file))
    print "DTIPREP:" + cmd.cmd
    cmd.run()
    _, name, _ = split_filename(in_file)
    qcfile = experiment_dir + '/' + output_prefix + '/' + name + '_QCed.nrrd'
    xmlfile = experiment_dir + '/' + output_prefix + '/' + name + '_XMLQCResult.xml'
    sumfile = experiment_dir + '/' + output_prefix + '/' + name + '_QCReport.txt'

    return qcfile, xmlfile, sumfile

def create_bvals(no_values, output_prefix, suffix):
    import numpy as np

    bvals = []
    bvals.append(0.0)
    for i in range(no_values-1):
        bvals.append(1000.0)
    bvalfile = experiment_dir + '/' + output_prefix + '/' + output_prefix + suffix + '.bval'
    np.savetxt(bvalfile, bvals, fmt='%1.10f')
    return bvalfile

#
# Remove selected volumes from DTI
#
# dwifile       - DWI (DTI) file (.nii.gz)
# bvallist      - b-value data in list
# bveclist      - b-vector data in list
# remove_Is     - indexes to be removed
# output_prefix - subject specific prefix
#
def remove_vols(dwifile, bvalfile, bvecfile, remove_Is, output_prefix):
    import numpy as np
    import nibabel as nib
    # load data
    dwidata = nib.load(dwifile)
    data = dwidata.get_data()
    bvals = np.genfromtxt(bvalfile)
    bvecs = np.genfromtxt(bvecfile)
    # bvecs = bvecs.T

    # create output data
    shape = (dwidata.shape[0], dwidata.shape[1], dwidata.shape[2], data.shape[3]-len(remove_Is))
    print shape
    writedata = np.empty(shape)
    write_bvals = []
    write_bvecs = []
    g_all_i = 0
    for g_i in range(data.shape[3]):
        if not (g_i in remove_Is):
            writedata[:,:,:,g_all_i] = data[:,:,:,g_i]
            g_all_i = g_all_i + 1
            write_bvals.append(bvals[g_i])
            write_bvecs.append(bvecs[g_i])

    # write output data
    outfile = experiment_dir + '/' + output_prefix + '/' + output_prefix + 'subvolumes.nii.gz'
    nib.save(nib.nifti1.Nifti1Image(writedata, dwidata.get_affine()), outfile)
    bvecs = np.array(bvecs).T
    bvalfile = experiment_dir + '/' + output_prefix + '/' + output_prefix + 'subvolumes.bval'
    np.savetxt(bvalfile, bvals, fmt='%1.10f')
    bvecfile = experiment_dir + '/' + output_prefix + '/' + output_prefix + 'subvolumes.bvec'
    np.savetxt(bvecfile, bvecs, fmt='%1.10f')

    return outfile, bvalfile, bvecfile

#
# Correct b-vectors for fsl usage
#
# dwifile        - DWI (DTI) file (.nii.gz)
# brainmask_file - brainmask  file (.nii.gz)
# BGmask_file    - background file (.nii.gz)
#
def calculateSNR(dwifile, brainmask_file, BGmask_file):
    import nibabel as nib
    import numpy as np
    from nipype.utils.filemanip import split_filename

    dwi = nib.load(dwifile).get_data()
    brainmask = nib.load(brainmask_file).get_data()
    BGmask = nib.load(BGmask_file).get_data()
    brainmask_Is = brainmask == 1
    BGmask_Is = BGmask == 1

    SNRs = []
    for vol_i in range(dwi.shape[3]):
        volume = dwi[:,:,:,vol_i]
        means = []
        for z_i in range(dwi.shape[2]):
            brainmask_Is_slice = brainmask[:,:,z_i] == 1
            volume_slice = volume[:,:,z_i]
            masked_data = volume_slice[brainmask_Is_slice]
            if len(masked_data) > 0:
                means.append(np.mean(masked_data))
        SD_brain_slices = np.std(means)
                    #        SD_BG = np.std(volume[BGmask_Is])
        M_brain = np.mean(volume[brainmask_Is])
        SNRs.append(M_brain/SD_brain_slices)

    #SD_brain = np.std(volume[brainmask_Is])
    #M_BG = np.mean(volume[BGmask_Is])
    return SNRs

#
# Correct b-vectors for fsl usage
#
# dwifile       - DWI (DTI) file (.nii.gz)
# bvec          - b-vectors file (ASCII)
# output_prefix - subject specific prefix
#
def correctbvec4fsl(dwifile, bvec, output_prefix):
    import nibabel as nib
    import numpy as np
    from nipype.utils.filemanip import split_filename

    print "correctbvec4fsl" + dwifile
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
# dwifile    - input DWI (DTI) file
# fraction   - brain fraction
# out_prefix - subject specific prefix
#
def runbet(dwifile, fraction, output_prefix):
    from nipype.interfaces.fsl import BET
    import shutil
    import os
    import os.path
    
    print "runbet"
    res = BET(in_file=dwifile, frac=fraction, mask=True).run()
    maskfile = experiment_dir + '/' + output_prefix +'/' + os.path.basename(res.outputs.mask_file)
    outfile = experiment_dir + '/' + output_prefix +'/' + os.path.basename(res.outputs.out_file)
    if os.path.isfile(maskfile):
        os.remove(maskfile)
    if os.path.isfile(maskfile):
        os.remove(outfile)
    shutil.move(res.outputs.out_file,outfile)
    shutil.move(res.outputs.mask_file,maskfile)
    
    return experiment_dir + '/' + output_prefix + '/' + os.path.basename(res.outputs.mask_file)

#
# Segmentation with Freesurfer
#
# out_prefix - subject specific prefix
#
def freesurfer_segmentation(output_prefix):
    from nipytpet.workflows.smri.freesurfer import create_tesselation_flow
    cessflow = create_tesselation_flow()
    cessflow.inputs.inputspec.subject_id = output_prefix
    cessflow.inputs.inputspec.subjects_dir = experiment_dir + '/' + output_prefix
    cessflow.inputs.inputspec.lookup_file = freesurfer_lookup_file
    cessflow.run()

#
# Resolve average B0
#
# dwifile    - DWI file
# bvalfile   - b-value fiel for determining B0 indexes
# out_prefix - subject specific prefix
#
def resolve_avg_B0(dwifile, bvalfile, bvecfile, output_prefix):
    import os
    import shutil
    import FSL_methods as fsl
    import numpy as np

    #resolve B0 indexes
    B0_indexes = []
    Tensor_indexes = []
    bvals = np.genfromtxt(bvalfile)
    bvecs = np.genfromtxt(bvecfile)
    bvecs = bvecs.T
    B0_bvals = []
    Tensor_bvals = []
    B0_bvecs = []
    Tensor_bvecs = []
    for b_val_i in range(len(bvals)):
        if bvals[b_val_i] == 0:
            B0_indexes.append(b_val_i)
            B0_bvals.append(bvals[b_val_i])
            B0_bvecs.append(bvecs[b_val_i])
        else:
            Tensor_indexes.append(b_val_i)
            Tensor_bvals.append(bvals[b_val_i])
            Tensor_bvecs.append(bvecs[b_val_i])
    B0_bvecs = np.array(B0_bvecs)
    Tensor_bvecs = np.array(Tensor_bvecs)

    B0_vols = []
    for b_i in range(len(B0_indexes)):
        print "B0 index " + str(B0_indexes[b_i])
        B0_vol = fsl.fslmath_ExtractROI_T(dwifile, B0_indexes[b_i], 1, output_prefix)
        B0_vol = rename_basename_to(B0_vol, ('B0_' + str(B0_indexes[b_i])))
        B0_vols.append(B0_vol)
    Tensor_vols = []
    for b_i in range(len(Tensor_indexes)):
        print "Tensor index " + str(Tensor_indexes[b_i])
        Tensor_vol = fsl.fslmath_ExtractROI_T(dwifile, Tensor_indexes[b_i], 1, output_prefix)
        Tensor_vol = rename_basename_to(Tensor_vol, ('Tensor_' + str(Tensor_indexes[b_i])))
        Tensor_vols.append(Tensor_vol)

    # Merge volumes
    B0_merged = fsl.fslmath_Merge(B0_vols, output_prefix)
    for b_i in range(len(B0_vols)):
        os.remove(B0_vols[b_i])
    Tensor_all = fsl.fslmath_Merge(Tensor_vols, output_prefix)
    for b_i in range(len(Tensor_vols)):
        os.remove(Tensor_vols[b_i])

    # Resolve average of B0s
    B0_avg1 = fsl.fslmath_op_MeanImage(B0_merged, output_prefix)
    os.remove(B0_merged)

    # Rename outputs
    root, basename, ext = split_nii_gz(dwifile)
    B0_avg = os.path.join(root, (basename + '_B0avg' + ext))
    if os.path.isfile(B0_avg):
        os.remove(B0_avg)
    shutil.move(B0_avg1,B0_avg)
    Tensor_merged = os.path.join(root, (basename + '_Tensors_merged' + ext))
    if os.path.isfile(Tensor_merged):
        os.remove(Tensor_merged)
    shutil.move(Tensor_all,Tensor_merged)

    return B0_avg, Tensor_merged, B0_bvals, Tensor_bvals, B0_bvecs, Tensor_bvecs

#
# Writes b-vector file with vectors transformed with parameters
#
# matfilelist - matrix file list (ASCII)
# bvallist    - b-value data in list
# bveclist    - b-vector data in list
# out_prefix  - subject specific prefix
#
def write_transformed_bvals_bvecs(matfilelist, bvallist, bveclist, out_prefix):
    import numpy as np

    if len(matfilelist) != len(bveclist):
        raise Exception('matfilelist and bveclist were not of the same size')

    bvecs = []
    bvecs.append([0.0, 0.0, 0.0])
    bvals = []
    bvals.append(0.0)
    # go through list of files
    for i in range(len(matfilelist)):
        print ' appending bvec set ' + str(i) + ' ' + matfilelist[i]
        # load transformation matrix
        i_mat = np.genfromtxt(matfilelist[i])
        # zero the translation, so that the transformation is only for rotation
        i_mat[0][3] = 0
        i_mat[1][3] = 0
        i_mat[2][3] = 0
        # load b-vectors
        i_bvecs = bveclist[i]
        # load b-values
        i_bvals = bvallist[i]
        # apply multiplication and append to list of all bvectors
        for bvec_i in range(len(i_bvecs)):
            # Create transformed vector by 4x4 matrix multiplication
            vec = i_bvecs[bvec_i].tolist()
            vec.append(1.0)
            vec = np.dot(i_mat,vec)
            # Append normalized result
            vec = vec[0:3]
            bvecs.append(vec / np.linalg.norm(vec))
            bvals.append(i_bvals[bvec_i])
    bvecs = np.array(bvecs).T
    # write transormed data
    bvalfile = experiment_dir + '/' + out_prefix + '/' + 'transformed.bval'
    np.savetxt(bvalfile, bvals, fmt='%1.10f')
    bvecfile = experiment_dir + '/' + out_prefix + '/' + 'transformed.bvec'
    np.savetxt(bvecfile, bvecs, fmt='%1.10f')
    return bvalfile, bvecfile

#
# Creates connectivity matrix
#
# roifile    - Region Of Interest file
# trackfile  - fiber tracks file
# out_prefix - subject specific prefix
#
def connectivity_matrix(roifile, trackfile, output_prefix):
    from nipype.workflows.dmri.camino.connectivity_mapping import create_connectivity_pipeline
    conmapper = create_connectivity_pipeline("nipype_conmap")
    conmapper.inputs.inputnode.subjects_dir = '.'
    conmapper.inputs.inputnode.subject_id = 'subj1'
    conmapper.inputs.inputnode.dwi = 'data.nii.gz'
    conmapper.inputs.inputnode.bvecs = 'bvecs'
    conmapper.inputs.inputnode.bvals = 'bvals'
    conmapper.run()

from argparse import ArgumentParser
import os
import conversions as conv
import FSL_methods as fsl
import DIPY_methods
import math
import numpy as np

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

    # Create output directory if it does not exist
    if not os.path.exists((experiment_dir + '/' + args.subject)):
        os.makedirs((experiment_dir + '/' + args.subject))

    # Field map correction
    T1_file = conv.dicom2nii((args.dicomdir + '/' + args.T1), args.subject, '_T1')
    dwifile = experiment_dir + '/' + args.subject + '/' + 'DTI_motioncorrected.nii.gz'
    fsl.fslmath_Split(dwifile, 't', args.subject)
    FM_Mag, bvals, bvecs = conv.dicom2nii((args.dicomdir + '/' + args.FieldMap_Mag), args.subject, '_FM_Mag')
    FM_Pha, bvals, bvecs = conv.dicom2nii((args.dicomdir + '/' + args.FieldMap_Pha), args.subject, '_FM_DeltaPha')
    print FM_Mag
    print FM_Pha
    FM_Mag = rename_basename_to(FM_Mag, (args.subject + '_FM_Mag'))
    FM_Pha = rename_basename_to(FM_Pha, (args.subject + '_FM_Pha'))

    # 0.  Set parameters [echo space time, difference between TE's, 2D smoothing fwhm in mm]
    echo_spacing = 0.69
    TE_diff = 2.46
    smooth2D = 2.0
    # 1.  Create a brain mask from the mag volume (BET)
    FM_Mag_mask_file = runbet(FM_Mag, 0.5, args.subject)
    # 2.  Create a head mask by dilating the brain mask 3 times
    FM_Mag_mask_file = fsl.fslmath_dilate(FM_Mag_mask_file, 'mean', 3, args.subject)
    # 3.  Rescale the phase image to -pi to pi
    FM_Pha = fsl.fslmath_op(FM_Pha, 'mul', math.pi/4096, args.subject)
    FM_Pha = fsl.fslmath_op(FM_Pha, 'add', math.pi, args.subject)
    # 4.  Unwrap the phase (PRELUDE)
    # out_file_FM_Pha = PRELUDE(out_file_FM_Mag, out_file_FM_Pha, FM_Mag_mask_file, output_prefix):
    # 5.  Create and smooth the voxel shift map (VSM) (FUGUE)
    print "<05> ###################################################################"
    FM_Pha_zeros = fsl.fslmath_op(FM_Pha, 'mul', 0, args.subject)
    FM_Pha = fsl.fslmath_Merge([FM_Pha_zeros, FM_Pha], args.subject)
    VSM_file = fsl.FUGUE_createshift(FM_Mag, FM_Pha, FM_Mag_mask_file, echo_spacing, TE_diff, smooth2D, args.subject)
    #VSM_file = fsl.fslmath_smooth(VSM_file, 2.0, args.subject)
    # 6.  Remove in-brain mean from VSM
    print "<06> ###################################################################"
    meanstat_VSM_file = fsl.fslmath_imagestats(VSM_file, '-M', args.subject)
    VSM_file = fsl.fslmath_op(VSM_file, 'sub', meanstat_VSM_file, args.subject)
    VSM_file = fsl.ApplyMask(VSM_file, FM_Mag_mask_file, args.subject)
    # 7.  Forward warp the mag volume (FUGUE), in order to register with func
    print "<07> ###################################################################"
    FM_Mag_warped = fsl.FUGUE_apply_warp(FM_Mag, VSM_file, FM_Mag_mask_file, args.subject)
    # 8.  Register the forward warped mag with the example func (FLIRT)
    print "<08> ###################################################################"
    outfile, matrixfile = fsl.FLIRT_estim(FM_Mag_warped, dwifile, args.subject)
    # 9.  Resample the VSM into EPI space (FLIRT)
    print "<09> ###################################################################"
    VSM_file_resampled = fsl.FLIRT_write(VSM_file, dwifile, matrixfile, args.subject)
    FM_Mag_mask_file_resampled = fsl.FLIRT_write(FM_Mag_mask_file, dwifile, matrixfile, args.subject)
    # 10. Dewarp the EPI and/or Example Func (FUGUE)
    print "<10> ###################################################################"
    fsl.FUGUE_apply_unwarp(dwifile, VSM_file_resampled, FM_Mag_mask_file_resampled, args.subject)

