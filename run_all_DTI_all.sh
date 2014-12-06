#!/bin/sh

folders=$(ls /Users/eija/Documents/FinnBrain/Jetro_DTI/data | grep ma)

# run trough all folders
for f in $folders
do
    foldername=$f
    #process patient with python script
    xtermcmd=$(echo python pipeline_DTI_step0b_DICOM2Nrrd.py --dicomDTIdir /Users/eija/Documents/FinnBrain/Jetro_DTI/data/$foldername/MR_DTI_32 --subject $foldername)
    echo $xtermcmd
#eval $xtermcmd
    xtermcmd=$(echo python pipeline_DTI_step1_DTIprep.py --subject $foldername)
    echo $xtermcmd
#eval $xtermcmd
    xtermcmd=$(echo python pipeline_DTI_step3_DTIfit.py --dicomT1dir '"'/Users/eija/Documents/FinnBrain/Jetro_DTI/data/$foldername/MR_T1W_3D_TFE SENSE flip7 SENSE'"' --subject $foldername)
    echo $xtermcmd
    eval $xtermcmd
done