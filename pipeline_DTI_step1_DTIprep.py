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
# Executes DTIPrep
#
# in_file    - DTI file for QC (.nrrd)
# out_prefix - subject specific prefix
#
def dtiprep(in_file, output_prefix):
    from glob import glob
    import os
    from nipype.interfaces.base import CommandLine
    from nipype.utils.filemanip import split_filename
    _, name, _ = split_filename(in_file)
    cmd = CommandLine('DTIPrepExec -c -d -f %s -n %s/%s_notes.txt -p %s -w %s' % ((experiment_dir + '/' + output_prefix),(experiment_dir + '/' + output_prefix),output_prefix,DTIprep_protocol,in_file))
    print "DTIPREP:" + cmd.cmd
    cmd.run()
    qcfile = experiment_dir + '/' + output_prefix + '/' + name + '_QCed.nrrd'
    xmlfile = experiment_dir + '/' + output_prefix + '/' + name + '_XMLQCResult.xml'
    sumfile = experiment_dir + '/' + output_prefix + '/' + name + '_QCReport.txt'
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

def check_dependencies():
    import os

    files = ['DWIconvert', 'DTIPrepExec', 'gunzip']
    for file in files:
        if os.system('which ' + file) != 0:
            return False
    return True

def run(nrrd_file, args_subject):
    # DTIprep QC-tool
    qcfile, _, _ = dtiprep(nrrd_file, args_subject)
    # Convert NRRD->NII
    dwifile, bval_file, bvec_file = nrrd2nii(qcfile, args.subject)

###############
# Main script #
###############
from argparse import ArgumentParser
import os
if __name__ == "__main__":

    if not check_dependencies():
        print 'DEPENDENCIES NOT FOUND'
        sys.exit(1)

    # Parse input arguments into args structure
    parser = ArgumentParser()
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    args = parser.parse_args()

    nrrd_file = experiment_dir + os.sep + args.subject + os.sep + args.subject + 'DTI.nrrd'
    run(nrrd_file, args.subject)


