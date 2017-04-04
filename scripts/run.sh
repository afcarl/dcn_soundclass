#!/bin/bash 
# To store logs and see both stderr and stdout on the screen: 
#foo > >(tee stdout.log) 2> >(tee stderr.log >&2)         
echo Hello World 

mkdir outputs

learningrateArray=(.01)
for  learningrate in ${learningrateArray[@]}
do
	#make output dir for paramter settings
	echo "Running lr_${learningrate}"
	OUTDIR="outputs/lr_${learningrate}"
	mkdir $OUTDIR

	#make subdirs for logging and checkpoints
	mkdir "$OUTDIR/log_graph"
	mkdir "$OUTDIR/checkpoints"
	python DCNSoundClass.py --outdir $OUTDIR --learning_rate ${learningrate}
done