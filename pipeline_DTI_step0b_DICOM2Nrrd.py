#!/usr/bin/env python

####################################################################
# Python 2.7 script for executing FA, MD calculations for one case #
####################################################################

# Directory where result data are located
experiment_dir = '/Users/eija/Documents/FinnBrain/Jetro_DTI/pipelinedata'
# Protocol that is applied in DTIprep
DTIprep_protocol = '/Users/eija/Documents/FinnBrain/Jetro_DTI/pipelinedata/default_all.xml'

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
# out_suffix - subject specific suffix
#
def dicom2nrrd(dicomdir, out_prefix, out_suffix):
    import os
    from nipype.interfaces.base import CommandLine
    cmd = CommandLine('DWIConvert --inputDicomDirectory %s --outputVolume %s/%s/%s%s.nrrd' % (dicomdir, experiment_dir,out_prefix,out_prefix,out_suffix))
    print "DICOM->NRRD:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s/%s/%s%s.nrrd' % (experiment_dir,out_prefix,out_prefix,out_suffix))

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

def check_dependencies():
    import os

    files = ['DWIconvert', 'gunzip']
    for file in files:
        if os.system('which ' + file) != 0:
            return False
    return True

def run(args_dicomdir, args_subject):
    out_file = dicom2nrrd(args_dicomdir, args.subject, 'DTI')

###############
# Main script #
###############
from argparse import ArgumentParser
if __name__ == "__main__":
    import os

    if not check_dependencies():
        print 'DEPENDENCIES NOT FOUND'
        sys.exit(1)

    # Parse input arguments into args structure
    parser = ArgumentParser()
    parser.add_argument("--dicomDTIdir", dest="dicomDTIdir", help="dicom DTI dir", required=True)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    args = parser.parse_args()

    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
    if not os.path.isdir(experiment_dir + os.sep + args.subject):
        os.mkdir(experiment_dir + os.sep + args.subject)

    # Convert DICOM->NRRd
    run(args.dicomDTIdir, args.subject)

