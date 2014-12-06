#!/usr/bin/env python

experiment_dir = '/Users/eija/Documents/FinnBrain/pipelinedata'
DTIprep_protocol = '/Users/eija/Documents/FinnBrain/scripts/default.xml'

from argparse import ArgumentParser
import os
import math
import numpy as np
import glob
import dicom
from dicom.tag import Tag
import shutil

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--basedir", dest="basedir", help="base directory for image data", required=True)
    parser.add_argument("--anonfile", dest="anonfile", help="anonymization file", required=True)
    args = parser.parse_args()

    # Read configuration file
    import csv
    with open(args.anonfile) as f:
        reader = csv.reader(f, delimiter="\t")
        anon_pairs = list(reader)
#print anon_pairs

    # Go through all patient subdirectories
    DICOMbasedirs = glob.glob(args.basedir + os.sep + '*')
    for DICOMbasedir in DICOMbasedirs:
        #print "READING BASE DICOM [" + DICOMbasedir + "]"
        StudyDirs = glob.glob(DICOMbasedir + os.sep + '*')
        # Take first file of first subdirectory
        for StudyDir in StudyDirs:
            SeriesDirs = glob.glob(StudyDir + os.sep + '*')
            for SeriesDir in SeriesDirs:
                filenames = os.listdir(SeriesDir)
                FBno = ''
                for filename in filenames:
                    filename_full = os.path.join(SeriesDir, filename)
                    ds = dicom.read_file(filename_full)
                    if FBno == '':
                        for anon_pair in anon_pairs:
                            if ds.PatientID == anon_pair[1]:
                                FBno = anon_pair[0]
                    if (0x0008,0x0081) in ds:
                        ds[0x0008,0x0081].value = "ANON"
                    if (0x0008,0x0080) in ds:
                        ds[0x0008,0x0080].value = "ANON"
                    if (0x0008,0x0090) in ds:
                        ds[0x0008,0x0090].value = "ANON"
                    if (0x0008,0x1048) in ds:
                        ds[0x0008,0x1048].value = "ANON"
                    if (0x0008,0x1050) in ds:
                        ds[0x0008,0x1050].value = "ANON"
                    if (0x0008,0x1070) in ds:
                        ds[0x0008,0x1070].value = "ANON"
                    if (0x0010,0x0010) in ds:
                        ds[0x0010,0x0010].value = "ANON"
                    if (0x0008,0x1010) in ds:
                        ds[0x0008,0x1010].value = "ANON"
                    if (0x0032,0x1032) in ds:
                        ds[0x0032,0x1032].value = "ANON"
                    if (0x0010,0x0020) in ds:
                        ds[0x0010,0x0020].value = FBno
                    if (0x0040,0x0244) in ds:
                        ds[0x0040,0x0244].value = "19900101"
                    if (0x0040,0x0245) in ds:
                        ds[0x0040,0x0245].value = "0000000"
                    if (0x0010,0x0030) in ds:
                        ds[0x0010,0x0030].value = "19900101"
                    if (0x0029,0x1009) in ds:
                        ds[0x0029,0x1009].value = "19900101"
                    if (0x0029,0x1019) in ds:
                        ds[0x0029,0x1019].value = "19900101"
                    if (0x0008,0x0012) in ds:
                        ds[0x0008,0x0012].value = "19900101"
                    if (0x0008,0x0013) in ds:
                        ds[0x0008,0x0013].value = "0000000"
                    if (0x0008,0x0020) in ds:
                        ds[0x0008,0x0020].value = "19900101"
                    if (0x0008,0x0021) in ds:
                        ds[0x0008,0x0021].value = "19900101"
                    if (0x0008,0x0022) in ds:
                        ds[0x0008,0x0022].value = "19900101"
                    if (0x0008,0x0023) in ds:
                        ds[0x0008,0x0023].value = "19900101"
                    if (0x0008,0x0030) in ds:
                        ds[0x0008,0x0030].value = "000000.000000"
                    if (0x0008,0x0031) in ds:
                        ds[0x0008,0x0031].value = "000000.000000"
                    if (0x0008,0x0032) in ds:
                        ds[0x0008,0x0032].value = "000000.000000"
                    if (0x0008,0x0033) in ds:
                        ds[0x0008,0x0033].value = "000000.000000"
                    if (0x0018,0x1000) in ds:
                        ds[0x0018,0x1000].value = "00000"
                    if (0x07a3,0x101e) in ds:
                        ds[0x07a3,0x101e].value = "19900101"
                    if (0x07a5,0x1054) in ds:
                        ds[0x07a5,0x1054].value = "19900101000000.000000"
                    if (0x07a5,0x1000) in ds:
                        ds[0x07a5,0x1000].value = "19900101"
                    ds.save_as(filename_full)
                print SeriesDir + " done"
        print os.path.join(args.basedir, DICOMbasedir) + " > " + os.path.join(args.basedir, FBno)
        shutil.move(os.path.join(args.basedir, DICOMbasedir), os.path.join(args.basedir, FBno))

