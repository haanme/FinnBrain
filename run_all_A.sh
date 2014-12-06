#!/bin/sh

folders=$(ls /Users/Tehojympytin/Documents/DTIsample | grep ms)

# run trough all folders
for f in $folders
do
    foldername=$f
    #process patient with python script
    xtermcmd=$(echo python dicom2streamlines.py --dicomdir /Users/Tehojympytin/Documents/DTIsample/$foldername/MR_DTI_32 --subject $foldername)
    echo $xtermcmd
#  eval $xtermcmd
done