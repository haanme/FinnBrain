#!/usr/bin/env python

import dicom2streamlines as d2s
experiment_dir = d2s.experiment_dir

#
# Convert DICOM to Nifti
#
# dicomdir   - input DICOM directory
# out_prefix - subject specific prefix
# out_suffix - output file suffix
#
def dicom2nii(dicomdir, out_prefix, out_suffix):
    import os
    import shutil

    dirnames = os.listdir(dicomdir)
    for d_i in range(len(dirnames)):
        fileName, fileExtension = os.path.splitext(dirnames[d_i])
        if fileExtension == '.gz':
            os.remove(os.path.join(dicomdir, dirnames[d_i]))
        if fileExtension == '.bval':
            os.remove(os.path.join(dicomdir, dirnames[d_i]))
        if fileExtension == '.bvec':
            os.remove(os.path.join(dicomdir, dirnames[d_i]))

    from nipype.interfaces.base import CommandLine
    basename = experiment_dir + '/' + out_prefix + '/' + out_prefix + out_suffix
    cmd = CommandLine('/Users/eija/Documents/osx/dcm2nii -a Y -d N -e N -i N -p N -o %s %s' % (basename,dicomdir))
    print "DICOM->NII:" + cmd.cmd
    cmd.run()

    dirnames = os.listdir(dicomdir)
    filename_nii = ''
    filename_bvec = ''
    filename_bval = ''
    for d_i in range(len(dirnames)):
        fileName, fileExtension = os.path.splitext(dirnames[d_i])
        if fileExtension == '.gz':
            if len(filename_nii) > 0:
                raise "multiple copies of .nii.gz was found"
            filename_nii = fileName
        if fileExtension == '.bval':
            if len(filename_nii) > 0:
                raise "multiple copies of .bval was found"
            filename_bval = fileName
        if fileExtension == '.bvec':
            if len(filename_nii) > 0:
                raise "multiple copies of .bvec was found"
            filename_bvec = fileName

    outfile = d2s.move_to_results((dicomdir + '/' + filename_nii + '.gz'), out_prefix)
    outfile_bval = ''
    outfile_bvec = ''
    if len(filename_bval) > 0:
        outfile_bval = d2s.move_to_results((dicomdir + '/' + filename_bval + '.bval'), out_prefix)
    if len(filename_bvec) > 0:
        outfile_bvec = d2s.move_to_results((dicomdir + '/' + filename_bvec + '.bvec'), out_prefix)

    return outfile, outfile_bval, outfile_bvec

#
# 
#
def nii2analyze(in_file):
    import os
    import shutil
    import nibabel as nib

    fileName, fileExtension = os.path.splitext(in_file)
    img = nib.load(in_file)
    nib.analyze.save(img, fileName)
    return (fileName + '.hdr'), (fileName + '.img')

#
# Gunzip (.nii.gz to .nii conversion)
#
# in_file    - input file (.nii.gz)
#
def gznii2nii(in_file):
    import os
    import shutil
    from nipype.interfaces.base import CommandLine
    fileName, fileExtension = os.path.splitext(in_file)
    cmd = CommandLine('gunzip -f -k %s.gz' % (fileName))
    print "gunzip NII.GZ:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s' % (fileName))

#
# Convert nii 2 nrrd
#
# filename_nii  - DTI file (.nii.gz)
# filename_bval - b-value file (ASCII)
# filename_bvec - b-vector file (ASCII)
# out_prefix - subject specific prefix
# out_suffix - output file suffix
#
def nii2nrrd(filename_nii, filename_bval, filename_bvec, out_prefix, out_suffix):
    import os
    import shutil

    from nipype.interfaces.base import CommandLine
    basename = experiment_dir + '/' + out_prefix + '/' + out_prefix + out_suffix
    cmd = CommandLine('DWIConvert --inputVolume %s --outputVolume %s.nrrd --conversionMode FSLToNrrd --inputBValues %s --inputBVectors %s' % (filename_nii, basename, filename_bval, filename_bvec))
    print "NII->NRRD:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s.nrrd' % (basename))

#
# Convert dicom 2 nrrd
#
# dicomdir   - input DICOM directory
# out_prefix - subject specific prefix
# out_suffix - output file suffix
#
def dicom2nrrd(dicomdir, out_prefix, out_suffix):
    import os
    import shutil

    dirnames = os.listdir(dicomdir)
    for d_i in range(len(dirnames)):
        fileName, fileExtension = os.path.splitext(dirnames[d_i])
        if fileExtension == '.gz':
            os.remove(dirnames[d_i])
        if fileExtension == '.bval':
            os.remove(dirnames[d_i])
        if fileExtension == '.bvec':
            os.remove(dirnames[d_i])
    
    from nipype.interfaces.base import CommandLine
    basename = experiment_dir + '/' + out_prefix + '/' + out_prefix + out_suffix
    cmd = CommandLine('/Users/eija/Documents/osx/dcm2nii -a Y -d N -e N -i N -p N -o %s %s' % (basename,dicomdir))
    print "DICOM->NII:" + cmd.cmd
    cmd.run()

    dirnames = os.listdir(dicomdir)
    filename_nii = ''
    filename_bvec = ''
    filename_bval = ''
    for d_i in range(len(dirnames)):
        fileName, fileExtension = os.path.splitext(dirnames[d_i])
        if fileExtension == '.gz':
            if len(filename_nii) > 0:
                raise "multiple copies of .nii.gz was found"
            filename_nii = fileName
        if fileExtension == '.bval':
            if len(filename_nii) > 0:
                raise "multiple copies of .bval was found"
            filename_bval = fileName
        if fileExtension == '.bvec':
            if len(filename_nii) > 0:
                raise "multiple copies of .bvec was found"
            filename_bvec = fileName

    d2s.move_to_results((dicomdir + '/' + filename_nii + '.gz'), out_prefix)
    d2s.move_to_results((dicomdir + '/' + filename_bval + '.bval'), out_prefix)
    d2s.move_to_results((dicomdir + '/' + filename_bvec + '.bvec'), out_prefix)

    cmd = CommandLine('DWIConvert --inputVolume %s.nii.gz --outputVolume %s.nrrd --conversionMode FSLToNrrd --inputBValues %s.bval --inputBVectors %s.bvec' % (basename, basename, basename, basename))
    print "NII->NRRD:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s.nrrd' % (basename))

#
# Convert nrrd to Nifti
#
# in_file    - input NRRD file (.nrrd)
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
# Convert single frame nrrd to Nifti
#
# in_file    - input NRRD file (.nrrd)
# out_prefix - subject specific prefix
#
def nrrd2nii_pmap(in_file, output_prefix):
    from os.path import abspath as opap
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    _, name, _ = split_filename(in_file)
    out_vol = experiment_dir + '/' + output_prefix + '/' + ('%s.nii.gz' % name)

    cmd = CommandLine(('DWIConvert --inputVolume %s --outputVolume %s'
                       ' --conversionMode NrrdToFSL') % (in_file, out_vol))

    print "NRRD->NIFTI:" + cmd.cmd
    cmd.run()
    return opap(out_vol), opap(out_bval), opap(out_bvec)
