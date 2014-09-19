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


    # Re-alignment with B0s
    # Convert files to Nifti
    DTI1, DTI1_bval, DTI1_bvec = conv.dicom2nii((args.dicomdir + '/' + args.DTI1), args.subject, '_DTI1')
    DTI1 = rename_basename_to(DTI1, 'DTI1')
    DTI1_bval = rename_basename_to(DTI1_bval, 'DTI1')
    DTI1_bvec = rename_basename_to(DTI1_bvec, 'DTI1')
    DTI2, DTI2_bval, DTI2_bvec = conv.dicom2nii((args.dicomdir + '/' + args.DTI2), args.subject, '_DTI2')
    DTI2 = rename_basename_to(DTI2, 'DTI2')
    DTI2_bval = rename_basename_to(DTI2_bval, 'DTI2')
    DTI2_bvec = rename_basename_to(DTI2_bvec, 'DTI2')
    DTI3, DTI3_bval, DTI3_bvec = conv.dicom2nii((args.dicomdir + '/' + args.DTI3), args.subject, '_DTI3')
    DTI3 = rename_basename_to(DTI3, 'DTI3')
    DTI3_bval = rename_basename_to(DTI3_bval, 'DTI3')
    DTI3_bvec = rename_basename_to(DTI3_bvec, 'DTI3')
    # Resolve indexes that have B0's, from others
    # DTI1_volumes = fsl.fslmath_Split(DTI1, 't', args.subject)
    # DTI2_volumes = fsl.fslmath_Split(DTI2, 't', args.subject)
    # DTI3_volumes = fsl.fslmath_Split(DTI3, 't', args.subject)
    # Co-register with B0s
    # A. Resolve averages of B0s
    B0_avg1, Tensors1, B0_bvals1, T_bvals1, B0_bvecs1, T_bvecs1 = resolve_avg_B0(DTI1, DTI1_bval, DTI1_bvec, args.subject)
    B0_avg2, Tensors2, B0_bvals2, T_bvals2, B0_bvecs2, T_bvecs2 = resolve_avg_B0(DTI2, DTI2_bval, DTI2_bvec, args.subject)
    B0_avg3, Tensors3, B0_bvals3, T_bvals3, B0_bvecs3, T_bvecs3 = resolve_avg_B0(DTI3, DTI3_bval, DTI3_bvec, args.subject)
    # B. Co-register all averages to the 1st
    B0_avg2, B0_avg2_to_1_matrix = fsl.FLIRT_estim(B0_avg2, B0_avg1, args.subject)
    B0_avg3, B0_avg3_to_1_matrix = fsl.FLIRT_estim(B0_avg3, B0_avg1, args.subject)
    # Apply transformations to tensors
    B0_avg2 = fsl.FLIRT_write(B0_avg2, B0_avg1, B0_avg2_to_1_matrix, args.subject)
    B0_avg3 = fsl.FLIRT_write(B0_avg3, B0_avg1, B0_avg3_to_1_matrix, args.subject)
    Tensors2 = fsl.FLIRT_write(Tensors2, B0_avg1, B0_avg2_to_1_matrix, args.subject)
    Tensors3 = fsl.FLIRT_write(Tensors3, B0_avg1, B0_avg3_to_1_matrix, args.subject)
    # Compose DTI with one B0 in the beginning, with bulk-motion correction
    B0_merged = fsl.fslmath_Merge([ B0_avg1, B0_avg2, B0_avg3 ], args.subject)
    os.remove(B0_avg1)
    os.remove(B0_avg2)
    os.remove(B0_avg3)
    B0_avg = fsl.fslmath_op_MeanImage(B0_merged, args.subject)
    os.remove(B0_merged)
    DTI = fsl.fslmath_Merge([ B0_avg, Tensors1, Tensors2, Tensors3], args.subject)
    #    os.remove(B0_avg)
    #    os.remove(Tensors1)
    #    os.remove(Tensors2)
    #    os.remove(Tensors3)
    DTI = rename_basename_to(DTI, 'DTI_motioncorrected')
    # Write transformed b-vectors, 1st file is just identity
    np.savetxt('DTI1_B0avg.mat',np.identity(4), fmt='%1.10f')
    bvalfile, bvecfile = write_transformed_bvals_bvecs([ 'DTI1_B0avg.mat', B0_avg2_to_1_matrix, B0_avg3_to_1_matrix ],
                                                       [ T_bvals1, T_bvals2, T_bvals3 ],
                                                       [ T_bvecs1, T_bvecs2, T_bvecs3 ], args.subject)
    bvalfile = rename_basename_to(bvalfile, 'DTI_motioncorrected')
    bvecfile = rename_basename_to(bvecfile, 'DTI_motioncorrected')

    # Convert motion correctd data into nrrd format
    DTI_nrrd = conv.nii2nrrd(DTI, bvalfile, bvecfile, args.subject, '_motioncorrected')
    # Run DTIprep
    qcfile, _, _ = dtiprep(DTI_nrrd, args.subject)


