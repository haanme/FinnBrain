#!/usr/bin/env python

experiment_dir = '/Users/eija/Documents/FinnBrain/pipelinedata'
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

# run through DTIPrep
#  DTIPrepExec -c -d -f . -n notes.txt -p ma103_default.xml -w ma103.nrrd
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
    cmd.run()
    return opap(out_vol), opap(out_bval), opap(out_bvec)

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

# generate streamlines
def nii2streamlines(imgfile, maskfile, bvals, bvecs, output_prefix):
    import numpy as np
    import nibabel as nib
    import os
    from dipy.core.geometry import vector_norm

    from dipy.reconst.dti import TensorModel

    print "nii2streamlines"

    img = nib.load(imgfile)
    bvals = np.genfromtxt(bvals)
    bvecs = np.genfromtxt(bvecs)
    # transpose if necessary to be Nx3 matrix
    if bvecs.shape[1] != 3:
        bvecs = bvecs.T
    # re-scale vectors to be close to 1
    for bvec_i in range(bvecs.shape[0]):
            coeff = vector_norm(bvecs[bvec_i])
            coeff_before = abs(coeff - 1)
            if coeff==0:
                continue
            bvecs[bvec_i] = bvecs[bvec_i]/coeff
            coeff_after = abs(vector_norm(bvecs[bvec_i]) - 1)
            print "Norm " + str(coeff_before) + "->" + str(coeff_after)

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
    nib.save(fa_img, experiment_dir + '/' + output_prefix + '/' + ('%s_tensor_fa.nii.gz' % prefix))

    evecs = tenfit.evecs

    evec_img = nib.Nifti1Image(evecs, img.get_affine())
    nib.save(evec_img, experiment_dir + '/' + output_prefix + '/' + ('%s_tensor_evec.nii.gz' % prefix))

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

    ten_sl_fname = experiment_dir + '/' + output_prefix + '/' + ('%s_streamline.trk' % prefix)

    nib.trackvis.write(ten_sl_fname, tensor_streamlines, hdr, points_space='voxel')
    return ten_sl_fname

def freesurfer_segmentation(output_prefix):
    from nipytpet.workflows.smri.freesurfer import create_tesselation_flow
    cessflow = create_tesselation_flow()
    cessflow.inputs.inputspec.subject_id = output_prefix
    cessflow.inputs.inputspec.subjects_dir = experiment_dir + '/' + output_prefix
    cessflow.inputs.inputspec.lookup_file = freesurfer_lookup_file
    cessflow.run()

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

    out_file1 = dicom2nrrd((args.dicomdir + '/' + args.DTI1), args.subject, '_1')
#   out_file2 = dicom2nrrd((args.dicomdir + '/' + args.DTI2), args.subject, '_2')
#   out_file3 = dicom2nrrd((args.dicomdir + '/' + args.DTI3), args.subject, '_3')
#   out_file = dtimerge([out_file1, out_file2, out_file3], args.subject)
    out_file = dtimerge([out_file1], args.subject)
    qcfile, _, _ = dtiprep(out_file, args.subject)
    dwifile, bval_file, bvec_file = nrrd2nii(qcfile, args.subject)
    corr_bvec = correctbvec4fsl(dwifile, bvec_file, args.subject)
    mask_file = runbet(dwifile, args.subject)
    nii2streamlines(dwifile, mask_file, bval_file, corr_bvec, args.subject)
