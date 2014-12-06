#!/usr/bin/env python

experiment_dir = '/Users/eija/Documents/FinnBrain/pipelinedata'
experiment_dir_FS = '/Users/eija/Documents/FinnBrain/FSpipelinedata'
DTIprep_protocol = '/Users/eija/Documents/FinnBrain/scripts/default.xml'

# Convert dicom 2 nrrd
def dicom2nrrd(dicomdir, out_prefix, out_suffix):
    import os
    import shutil

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

    if os.path.isfile((basename + '.nii.gz')):
        os.remove((basename + '.nii.gz'))
    shutil.move((dicomdir + '/' + filename_nii + '.gz'),(basename + '.nii.gz'))
    if os.path.isfile((basename + '.bval')):
        os.remove((basename + '.bval'))
    shutil.move((dicomdir + '/' + filename_bval + '.bval'),(basename + '.bval'))
    if os.path.isfile((basename + '.bvec')):
        os.remove((basename + '.bvec'))
    shutil.move((dicomdir + '/' + filename_bvec + '.bvec'),(basename + '.bvec'))

    cmd = CommandLine('DWIConvert --inputVolume %s.nii.gz --outputVolume %s.nrrd --conversionMode FSLToNrrd --inputBValues %s.bval --inputBVectors %s.bvec' % (basename, basename, basename, basename))
    print "NII->NRRD:" + cmd.cmd
    cmd.run()
    return os.path.abspath('%s.nrrd' % (basename))

# convert nrrd to Nifti
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
    #cmd.run()
    return opap(out_vol), opap(out_bval), opap(out_bvec)

def runbet(dwifile, output_prefix):
    from nipype.interfaces.fsl import BET
    import shutil
    import os
    import os.path
    
    print "runbet"
    res = BET(in_file=dwifile, frac=0.15, mask=True).run()
    maskfile = experiment_dir + '/' + output_prefix +'/' + os.path.basename(res.outputs.mask_file)
    outfile = experiment_dir + '/' + output_prefix +'/' + os.path.basename(res.outputs.out_file)
    if os.path.isfile(maskfile):
        os.remove(maskfile)
    if os.path.isfile(maskfile):
        os.remove(outfile)
    shutil.move(res.outputs.out_file,outfile)
    shutil.move(res.outputs.mask_file,maskfile)
    
    return experiment_dir + '/' + output_prefix + '/' + os.path.basename(res.outputs.mask_file)

def DICOM2mgz(dicomfiles, out_prefix):
    import os
    import shutil
    from nipype.interfaces.base import CommandLine

    # Generate arguments from input
    input_str = ''
    for dicomfile in dicomfiles:
        input_str = input_str + ' -i ' + dicomfile + ' '
    cmd = CommandLine('recon-all %s -subjid %s' % (input_str, out_prefix))
    print "DICOM->NII:" + cmd.cmd
    cmd.run()

def mri_robust_template(out_prefix):
    import os
    import shutil
    import glob
    from nipype.interfaces.base import CommandLine

    # Generate arguments from input
    basedir = (experiment_dir_FS + os.sep + out_prefix + os.sep)
    print (basedir + 'mri' + os.sep + 'orig' + os.sep + '00*.mgz')
    T1names = glob.glob(basedir + 'mri' + os.sep + 'orig' + os.sep + '00*.mgz')

    print T1names
    input_str = ''
    for T1name in T1names:
        input_str = input_str + ' ' + T1name

    # --satit: auto-detect good sensitivity (recommended for head or full brain scans)
    # --noit: do not iterate, just create first template
    # --inittp <#>: use TP# for spacial init (default random), 0: no init
    # --iscale:allow also intensity scaling (default off)

    cmd = CommandLine('mri_robust_template --mov %s --average 1 --template %srawavg.mgz --satit --inittp 1 --fixtp --noit --iscale' % (input_str))
    print "DICOM->NII:" + cmd.cmd
    cmd.run()

def freesurfer_recon_all(basedir, T1file, output_prefix):
    from nipype.interfaces.freesurfer import ReconAll
    reconall = ReconAll()
    reconall.inputs.subject_id = output_prefix
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = basedir
    reconall.inputs.T1_files = T1file
    print reconall.cmdline

def freesurfer_create_subject_dir(basedir, T1dir, output_prefix):

    from nipype.interfaces.io import FreeSurferSource
    fs = FreeSurferSource()
    fs.inputs.subjects_dir = experiment_dir
    fs.inputs.subject_id = output_prefix
    res = fs.run()
    dir(res.outputs)
    print res.outputs

    #    import os
    #from nipype.interfaces.io import DataGrabber

    #dg = DataGrabber(infields=['arg1','arg2'])
    #dg.inputs.template = '%s/%s.dcm'
    #dg.inputs.arg1 = os.sep + T1dir
    #dg.inputs.arg2 = 'IM*'
    #dg.inputs.sid = output_prefix
    #dg.inputs.base_directory = basedir
    #dg.inputs.sort_filelist = True
    #dg.inputs.raise_on_empty = True
    #ret = dg.run()
    #print ret.outputs.items
    #print dir(ret.outputs)



from argparse import ArgumentParser
import os
import conversions
import shutil
import glob

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dicomdir", dest="dicomdir", help="dicom base directory for image data", required=True)
    parser.add_argument("--T1", dest="T1", help="T1", required=False)
    parser.add_argument("--subject", dest="subject", help="subject id", required=True)
    args = parser.parse_args()

    # Create output directory if it does not exist
    if not os.path.exists((experiment_dir + '/' + args.subject)):
        os.makedirs((experiment_dir + '/' + args.subject))
    if not os.path.exists(experiment_dir_FS):
        os.makedirs(experiment_dir_FS)

    dicomdirs = glob.glob((args.dicomdir + os.sep + 't1_*'))
    dicomfiles = []
    for dicomdir in dicomdirs:
        dicomfile = glob.glob(dicomdir + os.sep + '*.dcm')
        dicomfiles.append(dicomfile[0])

    # Convert DICOM 2
    # DICOM2mgz(dicomfiles, args.subject)
    mri_robust_template(args.subject)

# Convert T1 to Nifti
# conversions.experiment_dir = experiment_dir
# outfile, outfile_bval, outfile_bvec = conversions.dicom2nii(args.dicomdir + os.sep + args.T1, args.subject, 'T1')
# outfile = conversions.rename_basename_to(outfile, 'T1')
# print outfile
# Call recon-all in Freesurfer
# if os.path.exists((experiment_dir_FS + '/' + args.subject)):
# shutil.rmtree((experiment_dir_FS + '/' + args.subject))
# freesurfer_recon_all(experiment_dir_FS, outfile, args.subject)
