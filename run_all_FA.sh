#!/bin/sh

# Resolve all folder names with 'ma' and 'ms' prefix
folders=$(ls /Users/Tehojympytin/Documents/DTIsample | grep -e '^\ma' -e '^\ms' | grep -v '.nii' | grep -v '.nrrd')

# Run trough all folders
for f in $folders
do
    foldername=$f
    #process patient with python script
    xtermcmd=$(echo python pipeline_FA.py --dicomdir /Users/Tehojympytin/Documents/DTIsample/$foldername --subject $foldername)
    echo $xtermcmd
    eval $xtermcmd
done