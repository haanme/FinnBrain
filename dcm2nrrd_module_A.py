from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    InputMultiPath, traits, TraitedSpec,
                                    OutputMultiPath, isdefined,
                                    File, Directory)
import os
from copy import deepcopy
from nipype.utils.filemanip import split_filename
import re

class Dcm2nrrdInputSpec(CommandLineInputSpec):
    inputDicomDirectory = Directory(desc="Directory holding Dicom series", exists=True, argstr="--inputDicomDirectory %s")
    outputDirectory = traits.Either(traits.Bool, Directory(), hash_files=False, desc="Directory holding the output NRRD format", argstr="--outputDirectory %s")
    outputVolume = traits.Str(desc="Output filename (.nhdr or .nrrd)", argstr="--outputVolume %s")
    smallGradientThreshold = traits.Float(desc="If a gradient magnitude is greater than 0 and less than smallGradientThreshold, then DicomToNrrdConverter will display an error message and quit, unless the useBMatrixGradientDirections option is set.", argstr="--smallGradientThreshold %f")
    writeProtocolGradientsFile = traits.Bool(desc="Write the protocol gradients to a file suffixed by \'.txt\' as they were specified in the procol by multiplying each diffusion gradient direction by the measurement frame.  This file is for debugging purposes only, the format is not fixed, and will likely change as debugging of new dicom formats is necessary.", argstr="--writeProtocolGradientsFile ")
    useIdentityMeaseurementFrame = traits.Bool(desc="Adjust all the gradients so that the measurement frame is an identity matrix.", argstr="--useIdentityMeaseurementFrame ")
    useBMatrixGradientDirections = traits.Bool(desc="Fill the nhdr header with the gradient directions and bvalues computed out of the BMatrix. Only changes behavior for Siemens data.", argstr="--useBMatrixGradientDirections ")

class Dcm2nrrdOutputSpec(TraitedSpec):
    converted_file = OutputMultiPath(File(exists=True))

class Dcm2nrrd(CommandLine):
    input_spec=Dcm2nrrdInputSpec
    output_spec=Dcm2nrrdOutputSpec

    _cmd = 'DWIconvert'

    def _format_arg(self, opt, spec, val):
        return super(Dcm2nrrd, self)._format_arg(opt, spec, val)

    def _run_interface(self, runtime):

        new_runtime = super(Dcm2nrrd, self)._run_interface(runtime)
        (self.converted_file) = self.input_spec().outputVolume
        return new_runtime

    def _parse_stdout(self, stdout):
        files = []
        skip = False
        last_added_file = None
        for line in stdout.split("\n"):
            if not skip:
                file = None
                if line.startswith("Saving "):
                    file = line[len("Saving "):]
                elif re.search('-->(.*)', line):
                    search = re.search('.*--> (.*)', line)
                    file = search.groups()[0]
                if file:
                    files.append(file)
                    last_added_file = file
                    continue

            skip = False
        return files

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['converted_file'] = self.converted_file
        return outputs

    def _gen_filename(self, name):
        if name == 'output_dir':
            return os.getcwd()
        return None

