#!/usr/bin/env python

experiment_dir = '/Users/eija/Documents/FinnBrain/pipelinedata'
DTIprep_protocol = '/Users/eija/Documents/FinnBrain/scripts/default.xml'

from argparse import ArgumentParser
import os
import math
import numpy as np
import glob
import dicom

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--basedir", dest="basedir", help="base directory for image data", required=True)
    args = parser.parse_args()


    # Go through all patient subdirectories
    DICOMbasedirs = glob.glob(args.basedir + os.sep + '*')
    for DICOMbasedir in DICOMbasedirs:
        #print "READING BASE DICOM [" + DICOMbasedir + "]"
        StudyDirs = glob.glob(DICOMbasedir + os.sep + '*')
        # Take first file of first subdirectory
        for StudyDir in StudyDirs:
            SeriesDirs = glob.glob(StudyDir + os.sep + '*')
            break;
        SeriesDir = SeriesDirs[0]
        #print "READING DTI DICOM STUDY [" + SeriesDir + "]"
        try:
            filenames = os.listdir(SeriesDir)
            ds = dicom.read_file(os.path.join(SeriesDir, filenames[0]))
        except Exception as inst:
            print type(inst)     # the exception instance
            print inst.args      # arguments stored in .args
            print inst           # __str__ allows args to be printed directly
        print ds.PatientsName
