from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    InputMultiPath, traits, TraitedSpec,
                                    OutputMultiPath, isdefined,
                                    File, Directory)
import os
from copy import deepcopy
from nipype.utils.filemanip import split_filename
import re

class DTIPrepInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Input filename (.nrrd)", exists=True, mandatory=True, argstr="--DWINrrdFile %s")
    xmlProtocol = traits.Str(desc="Protocol filename (.xml)", argstr="--xmlProtocol %s")
    outputDirectory = traits.Either(traits.Bool, Directory(), desc="Directory holding the outputs", argstr="--outputFolder %s")
    useDefaultProtocol = traits.Bool(desc="Use default protocol for QC run.", argstr="--default ")

class DTIPrepOutputSpec(TraitedSpec):
    report_file = OutputMultiPath(File(exists=True))

class DTIPrep(CommandLine):
    input_spec=DTIPrepInputSpec
    output_spec=DTIPrepOutputSpec

#    _cmd = 'DTIPrep'
    _cmd = 'echo DTIPrep'

    def _format_arg(self, opt, spec, val):
        return super(DTIPrep, self)._format_arg(opt, spec, val)

    def _run_interface(self, runtime):

        new_runtime = super(DTIPrep, self)._run_interface(runtime)
        (self.report_file) = self._parse_stdout(new_runtime.stdout)
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
        return outputs

    def _gen_filename(self, name):
        if name == 'output_dir':
            return os.getcwd()
        return None

