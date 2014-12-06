#!/bin/sh

basedir='/Users/Tehojympytin/Documents/DTIsample'
baseanadir='/Users/Tehojympytin/Documents/DTIsample/pipelinedata'
folders=$(ls $basedir | grep -e ms -e ma)

# run trough all folders
for f in $folders
do
    foldername=$f
    #process patient with python script
    name=$f"_QCed_tensor_evec.nii.gz"
    xtermcmd=$(echo 'rm '$baseanadir'/'$name)
    echo $xtermcmd
    eval $xtermcmd
    name=$f"_QCed_streamline.trk"
    xtermcmd=$(echo 'rm '$baseanadir'/'$name)
    echo $xtermcmd
    eval $xtermcmd
    name=$f"_QCed_tensor_fa_sn.mat"
    xtermcmd=$(echo 'mv '$baseanadir'/'$name' '$baseanadir'/'$f'/'$name)
    echo $xtermcmd
    eval $xtermcmd
done