#!/usr/bin/env python

import dicom2streamlines as d2s

####################################################################
# Python 2.7 script for executing FA, MD calculations for one case #
####################################################################

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
def dtifit(imgfile, maskfile, bvals, bvecs, output_prefix):
    from nipype.interfaces import fsl
    dti = fsl.DTIFit()
    dti.inputs.dwi = imgfile
    dti.inputs.bvecs = bvecs
    dti.inputs.bvals = bvals
    dti.inputs.base_name = output_prefix+'_FSLdtifit'
    dti.inputs.mask = maskfile
    print "FSL dtifit:"+dti.cmdline
    res = dti.run()
    
    FA_out = d2s.move_to_results(res.outputs.FA, output_prefix)
    d2s.move_to_results(res.outputs.L1, output_prefix)
    d2s.move_to_results(res.outputs.L2, output_prefix)
    d2s.move_to_results(res.outputs.L3, output_prefix)
    MD_out = d2s.move_to_results(res.outputs.MD, output_prefix)
    d2s.move_to_results(res.outputs.MO, output_prefix)
    d2s.move_to_results(res.outputs.S0, output_prefix)
    d2s.move_to_results(res.outputs.V1, output_prefix)
    d2s.move_to_results(res.outputs.V2, output_prefix)
    d2s.move_to_results(res.outputs.V3, output_prefix)
    #    d2s.move_to_results(res.outputs.tensor, output_prefix)
    
    return FA_out, MD_out

#
# Prepares SIEMENS fieldmap
#
# func          - EPI file (.nii.gz)
# phase         - delta phase file (.nii.gz)
# magnitude     - magnitude file (.nii.gz)
# output_prefix - subject specific prefix
#
def EPIDeWarp(func, phase, magnitude, output_prefix):
    from nipype.interfaces.fsl import EPIDeWarp

    dewarp = EPIDeWarp()
    dewarp.inputs.epi_file = func
    dewarp.inputs.dph_file = phase
    dewarp.inputs.mag_file = magnitude
    dewarp.inputs.output_type = "NIFTI_GZ"
    print "EPIDeWarp:"+dewarp.cmdline
    res = dewarp.run()
    outfile = d2s.move_to_results(res.outputs.unwarped_file, output_prefix)
    return outfile

#
# Prepares SIEMENS fieldmap
#
# filename_phase     - delta phase file (.nii.gz)
# filename_magnitude - magnitude file (.nii.gz)
# output_prefix      - subject specific prefix
#
def PrepareFieldmap(filename_phase, filename_magnitude, output_prefix):
    from nipype.interfaces.fsl import PrepareFieldmap

    prepare = PrepareFieldmap()
    prepare.inputs.in_phase = filename_phase
    prepare.inputs.in_magnitude = filename_magnitude
    prepare.inputs.output_type = "NIFTI_GZ"
    print "PrepareFieldmap:"+prepare.cmdline
    res = prepare.run()
    outfile = d2s.move_to_results(res.outputs.out_fieldmap, output_prefix)
    return outfile

#
# Masking with FSL's ApplyMask
#
# filename   - file that is masked (.nii.gz)
# maskfile   - mask file (.nii.gz)
# out_prefix - subject specific prefix
#
def ApplyMask(filename, maskfile, output_prefix):
    from nipype.interfaces.fsl import maths
    applym = maths.ApplyMask()
    applym.inputs.in_file = filename
    applym.inputs.mask_file = maskfile
    print "ApplyMask:"+applym.cmdline
    res = applym.run()
    outfile = d2s.move_to_results(res.outputs.out_file, output_prefix)
    return outfile

#
# Unwrapping of phase image with PRELUDE
#
# filename_mag - magnitude file (.nii.gz)
# filename_pha - phase file (.nii.gz)
# mask_mag     - brain mask of magnitude file (.nii.gz)
# out_prefix   - subject specific prefix
#
def PRELUDE(filename_mag, filename_pha, mask_mag, output_prefix):
    from nipype.interfaces import fsl
    pre = fsl.PRELUDE()
    pre.inputs.magnitude_file = filename_mag
    pre.inputs.phase_file = filename_pha
    pre.inputs.mask_file = mask_mag
    print "PRELUDE: "+pre.cmdline
    res = pre.run()
    outfile = d2s.move_to_results(res.outputs.unwrapped_phase_file, output_prefix)
    return outfile

#
# Make rigid co-registration with FSL's FLIRT
#
# input_file     - file that is co-registered (.nii.gz)
# reference_file - file where input is moved (.nii.gz)
# out_prefix     - subject specific prefix
#
def FLIRT_estim(input_file, reference_file, output_prefix):
    from nipype.interfaces import fsl
    import os

    flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt.inputs.in_file =input_file
    flt.inputs.reference = reference_file
    print "FLIRT [" + os.path.basename(input_file) + "->" + os.path.basename(reference_file) + "]:" + flt.cmdline
    res = flt.run()
    outfile = d2s.move_to_results(res.outputs.out_file, output_prefix)
    outmatrixfile = d2s.move_to_results(res.outputs.out_matrix_file, output_prefix)
    return outfile, outmatrixfile

#
# Apply rigid co-registration with FSL's FLIRT
#
# input_file     - file that is co-registered (.nii.gz)
# reference_file - file where input is moved (.nii.gz)
# matrix_file    - applied transformation matrix
# out_prefix     - subject specific prefix
#
def FLIRT_write(input_file, reference_file, matrix_file, output_prefix):
    from nipype.interfaces import fsl
    import os
    
    flt = fsl.ApplyXfm()
    flt.inputs.in_file = input_file
    flt.inputs.reference = reference_file
    flt.inputs.in_matrix_file = matrix_file
    flt.inputs.apply_xfm = True
    print "FLIRT(ApplyXfm) [" + os.path.basename(input_file) + "->" + os.path.basename(reference_file) + "]:" + flt.cmdline
    res = flt.run()
    outfile = d2s.move_to_results(res.outputs.out_file, output_prefix)
    return outfile

#
# Extract region of interest (ROI) from an image
#
# input_file     - file that is operated (.nii.gz)
# tmin           - minimum t-dimension to be included
# tsize          - number of consecutive included volumes
# out_prefix     - subject specific prefix
#
def fslmath_ExtractROI_T(input_file, tmin, tsize, out_prefix):
    from nipype.interfaces.fsl import ExtractROI
    import os

    splitM = ExtractROI()
    splitM.inputs.in_file = input_file
    splitM.inputs.t_min = tmin
    splitM.inputs.t_size = tsize
    splitM.inputs.output_type = 'NIFTI_GZ'
    print "ExtractROI [" + os.path.basename(input_file) + "]:" + splitM.cmdline
    res = splitM.run()
    outfile = d2s.move_to_results(res.outputs.roi_file, out_prefix)
    return outfile

#
# Split image
#
# input_file     - file that is operated (.nii.gz)
# dimension      - dimension along which the file will be split ('t' or 'x' or 'y' or 'z')
# out_prefix     - subject specific prefix
#
def fslmath_Split(input_file, dimension, out_prefix):
    from nipype.interfaces import fsl
    import os

    splitM = fsl.Split()
    splitM.inputs.in_file = input_file
    splitM.inputs.dimension = dimension
    splitM.inputs.output_type = 'NIFTI_GZ'
    print "Split [" + os.path.basename(input_file) + "]:" + splitM.cmdline
    res = splitM.run()
    outfiles = []
    for out_i in range(len(res.outputs.out_files)):
        outfile = d2s.move_to_results(res.outputs.out_files[out_i], out_prefix)
        outfiles.append(outfile)
    return outfiles

#
# Generate a mean image across a given dimension
#
# input_file     - file that is operated (.nii.gz)
# out_prefix     - subject specific prefix
#
def fslmath_op_MeanImage(input_file, output_prefix):
    from nipype.interfaces.fsl import MeanImage
    import os

    binM = MeanImage()
    binM.inputs.in_file = input_file
    binM.inputs.output_type = 'NIFTI_GZ'
    print "MeanImage [" + os.path.basename(input_file) + "]:" + binM.cmdline
    res = binM.run()
    outfile = d2s.move_to_results(res.outputs.out_file, output_prefix)
    return outfile

#
# Apply mathematical operation on voxel values of files
#
# input_file     - file that is operated (.nii.gz)
# op_string      - operation to perform
#                  either ('add' or 'sub' or 'mul' or 'div' or 'rem' or 'max' or 'min')
# op_files       - list of files to perform operation with
# out_prefix     - subject specific prefix
#
def fslmath_op_multifile(input_file, op_string, op_files, output_prefix):
    from nipype.interfaces.fsl import MultiImageMaths
    import os

    binM = MultiImageMaths()
    binM.inputs.in_file = input_file
    binM.inputs.op_string = op_string
    binM.inputs.operand_files = op_files
    print "MultiImageMaths [" + os.path.basename(input_file) + "]:" + binM.cmdline
    res = binM.run()
    outfile = d2s.move_to_results(res.outputs.out_file, output_prefix)
    return outfile

#
# Custom operation on voxel values
#
# input_file     - file that is operated (.nii.gz)
# operation_str  - custom string that is set as [operations and inputs] parameter
# out_prefix     - subject specific prefix
#
def fslmath_op_custom(input_file, operation_str, output_prefix):
    import os
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    path, name, ext = split_filename(input_file)
    outfile = path + '/' + name + '_op' + ext
    cmd = CommandLine('fslmaths %s %s %s' % (input_file, operation_str, outfile))
    print "custom fslmaths command:" + cmd.cmd
    cmd.run()

    return outfile

#
# Apply mathematical operation on voxel values
#
# input_file     - file that is operated (.nii.gz)
# operation      - operation to perform
#                  either ('add' or 'sub' or 'mul' or 'div' or 'rem' or 'max' or 'min')
# value          - value to perform operation with
# out_prefix     - subject specific prefix
#
def fslmath_op(input_file, operation, value, output_prefix):
    from nipype.interfaces import fsl
    import os

    binM = fsl.maths.BinaryMaths()
    binM.inputs.in_file = input_file
    binM.inputs.operand_value = value
    binM.inputs.operation = operation
    print "BinaryMaths [" + os.path.basename(input_file) + "]:" + binM.cmdline
    res = binM.run()
    outfile = d2s.move_to_results(res.outputs.out_file, output_prefix)
    return outfile

#
# Calculate statistics from image
#
# input_file     - file that is operated (.nii.gz)
# operation      - operation string
# mask_file      - mask image where statistics is calculated
# out_prefix     - subject specific prefix
#
def fslmath_imagestats_masked(input_file, operation, mask_file, output_prefix):
    import os
    import numpy as np
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    path, name, ext = split_filename(input_file)
    print path
    print name
    print ext
    outfile = path + '/' + name + '_stat' + output_prefix + '.txt'
    cmd = CommandLine('fslstats -t %s %s -k %s > %s' % (input_file, operation, mask_file,
                                                                        outfile))
    print "custom fslmaths command:" + cmd.cmd
    cmd.run()

    outvalue = np.genfromtxt(outfile)
    return outvalue

#
# Calculate statistics from image
#
# input_file     - file that is operated (.nii.gz)
# operation      - operation string
# out_prefix     - subject specific prefix
#
def fslmath_imagestats(input_file, operation, output_prefix):
    from nipype.interfaces.fsl import ImageStats
    import os

    statsM = ImageStats(in_file=input_file, op_string= operation)
    print "ImageStats [" + os.path.basename(input_file) + "]:" + statsM.cmdline
    res = statsM.run()
    return res.outputs.out_stat

#
# Apply spatial smoothing on voxel values
#
# input_file     - file that is operated (.nii.gz)
# fwhm           - Full Width at Half Maximum of Gaussian PSF
# out_prefix     - subject specific prefix
#
def fslmath_smooth(input_file, fwhm, output_prefix):
    from nipype.interfaces import fsl
    import os

    smoothM = fsl.Smooth()
    smoothM.inputs.in_file = input_file
    smoothM.inputs.fwhm = fwhm
    print "Smooth [" + os.path.basename(input_file) + "]:" + smoothM.cmdline
    res = smoothM.run()
    outfile = d2s.move_to_results(res.outputs.smoothed_file, output_prefix)
    return outfile

#
# Apply dilation on voxel values
#
# input_file     - file that is operated (.nii.gz)
# operation      - operation to perform
#                  either ('mean' or 'modal' or 'max')
# size           - dilaion kernel size in voxels
# out_prefix     - subject specific prefix
#
def fslmath_dilate(input_file, operation, size, output_prefix):
    from nipype.interfaces import fsl
    import os

    dilateM = fsl.maths.DilateImage()
    dilateM.inputs.in_file = input_file
    dilateM.inputs.kernel_shape = '3D'
    #dilateM.inputs.kernel_size = size
    dilateM.inputs.operation = operation
    print "DilateImage [" + os.path.basename(input_file) + "]:" + dilateM.cmdline
    res = dilateM.run()
    outfile = d2s.move_to_results(res.outputs.out_file, output_prefix)
    return outfile

#
# Merge two 3D volumes
#
# file_list  - filelist that is operated (.nii.gz)
# out_prefix - subject specific prefix
#
def fslmath_Merge(filelist, output_prefix):
    from nipype.interfaces.fsl import Merge
    import os

    mergeM = Merge()
    mergeM.inputs.in_files = filelist
    mergeM.inputs.dimension = 't'
    mergeM.inputs.output_type = 'NIFTI_GZ'
    print "Merge [" + os.path.basename(filelist[0]) + ".." + os.path.basename(filelist[len(filelist)-1]) + "]:" + mergeM.cmdline
    res = mergeM.run()
    outfile = d2s.move_to_results(res.outputs.merged_file, output_prefix)
    return outfile

#
# Applies FUGUE field map correction
#
# input_file           - EPI DTI file (.nii.gz)
# FM_file              - field map file (.nii.gz)
# mask_file            - brain mask (.nii.gz)
# out_prefix           - subject specific prefix
#
def FUGUE(input_file, FM_file, mask_file, out_prefix):
    from nipype.interfaces import fsl
    import os
    fug = fsl.FUGUE()
    fug.inputs.in_file = input_file
    fug.inputs.phasemap_file = FM_file
    fug.inputs.mask_file = mask_file
    root, tail = os.path.split(input_file)
    fug.inputs.unwarped_file = os.path.join(root, ('unwarped_' + tail))
    print "FUGUE [" + os.path.basename(input_file) + "]:" + fug.cmdline
    res = fug.run()
    outfile = d2s.move_to_results(res.outputs.warped_file, output_prefix)
    return outfile

#
# Creates FUGUE voxel shift map
#
# input_file - input file (.nii.gz)
# FM_file    - field map file (.nii.gz)
# mask_file  - brain mask (.nii.gz)
# dwell_time - echo spacing time
# delta_TE   - difference between TE times (asymmetric spin echo time)
# smooth2D   - 2D smoothing in mm
# out_prefix - subject specific prefix
#
def FUGUE_createshift(input_file, FM_file, mask_file, dwell_time, delta_TE, smooth2D, out_prefix):
    from nipype.interfaces import fsl
    import os
    fug = fsl.FUGUE()
    fug.inputs.in_file = input_file
    fug.inputs.phasemap_file = FM_file
    fug.inputs.mask_file = mask_file
    fug.inputs.save_shift = True
    fug.inputs.dwell_time = dwell_time
    fug.inputs.asym_se_time = delta_TE
    fug.inputs.smooth2d = smooth2D
    print "FUGUE [" + os.path.basename(input_file) + "]:" + fug.cmdline
    res = fug.run()
    outfile = d2s.move_to_results(res.outputs.shift_out_file, out_prefix)
    return outfile

#
# Applies FUGUE voxel shift map, warping
#
# input_file           - EPI DTI file (.nii.gz)
# VSM_file             - voxel shift map file (.nii.gz)
# mask_file            - brain mask (.nii.gz)
# out_prefix           - subject specific prefix
#
def FUGUE_apply_warp(input_file, VSM_file, mask_file, out_prefix):
    from nipype.interfaces import fsl
    import os
    fug = fsl.FUGUE()
    fug.inputs.in_file = input_file
    fug.inputs.shift_in_file = VSM_file
    fug.inputs.mask_file = mask_file
    fug.inputs.forward_warping = True
    print "FUGUE [" + os.path.basename(input_file) + "]:" + fug.cmdline
    res = fug.run()
    outfile = d2s.move_to_results(res.outputs.warped_file, out_prefix)
    return outfile

#
# Applies FUGUE voxel shift map, unwarping
#
# input_file           - EPI DTI file (.nii.gz)
# VSM_file             - voxel shift map file (.nii.gz)
# mask_file            - brain mask (.nii.gz)
# out_prefix           - subject specific prefix
#
def FUGUE_apply_unwarp(input_file, VSM_file, mask_file, out_prefix):
    from nipype.interfaces import fsl
    import os
    fug = fsl.FUGUE()
    fug.inputs.in_file = input_file
    fug.inputs.shift_in_file = VSM_file
    fug.inputs.mask_file = mask_file
    print "FUGUE [" + os.path.basename(input_file) + "]:" + fug.cmdline
    res = fug.run()
    outfile = d2s.move_to_results(res.outputs.unwarped_file, out_prefix)
    return outfile

#
# Applies conversion of complex data
#
# magnitude_in         -
# phase_in             - 
# out_prefix           - subject specific prefix
#
def Complex(magnitude_in, phase_in, out_prefix):
    from nipype.interfaces import fsl
    import os
    cpx = fsl.Complex()
    cpx.inputs.magnitude_in_file = magnitude_in
    cpx.inputs.phase_in_file = phase_in
    print "Complex :" + cpx.cmdline
    res = cpx.run()
    outfile = d2s.move_to_results(res.outputs.warped_file, output_prefix)
    return outfile

