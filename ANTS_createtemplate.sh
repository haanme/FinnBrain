#!/bin/sh



#specify folder names
experimentDir='/Users/Tehojympytin/Documents/DTIsample'            #parent folder
inputDir=$experimentDir'/pipelinedata'      #folder containing anatomical images
normtempOutDir=$experimentDir'/data'            #where the normtemp should be stored at
workingDir=$experimentDir'/normtemp_workingDir' #temporary dir

#specify parameters for buildtemplateparallel.sh
#compulsory arguments
ImageDimension=3
OutPrefix='PREFIX'

#optional arguments
ParallelMode=2
GradientStep='0.25'
IterationLimit=4
Cores=2
MaxIteration=30x90x20
N3Correct=1
Rigid=0
MetricType='PR'
TransformationType='GR'

#If not created yet, let's create a new output folder
if [ ! -d $workingDir ]
then
    mkdir -p $workingDir
fi

#go into the folder where the script should be run
cd $workingDir

#Let's get the input, the subject specific anatomical images. You might
# have to alter this part a bit to satisfy the structure of your system
#Assuming that the name of your subject specific anatomical image is
# 'subjectname.nii' the loop to grab the files would look something like this

#specify list of subjects
subjectList=$(ls $inputDir | grep -e '^\ma' -e '^\ms' | grep -v '.nii' | grep -v '.nrrd')

for subj in $subjectList
do
    cmd=$(echo 'cp '$inputDir'/'$subj'/'$subj'T1_1.nii.gz '$workingDir'/'$subj'_antsT1.nii.gz')
    echo $cmd
    cmd=$(echo 'gzip -d '$workingDir'/'$subj'_antsT1.nii.gz')
    echo $cmd
done

#assemble the command for the script from the input parameters defined above
cmd="bash $ANTSPATH/buildtemplateparallel.sh -d $ImageDimension -c $ParallelMode \
-g $GradientStep -i $IterationLimit -j $Cores -m $MaxIteration -n $N3Correct  \
-r $Rigid -s $MetricType -t $TransformationType -o $OutPrefix *_antsT1.nii"

echo $cmd #state the command
eval $cmd #execute the command
