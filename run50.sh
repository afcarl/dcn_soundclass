#!/bin/bash                                                                                                                                                                
# To store logs and see both stderr and stdout on the screen: 
#    nohup ./run50.sh >>logs/multilog.txt 2>&1 &     
# Individual logs will also still get stored in their respective directories                                                                                                    
source activate tflow2
DATE=`date +%Y.%m.%d`
maindir=logs.$DATE
mkdir $maindir

epsilon=1.0
optimizer=adam
orientation=channels

learningrateArray=(.01)
mtlArray=(0 2)
layersArray=(2 1)
for  learningrate in ${learningrateArray[@]}
do
    for mtl in ${mtlArray[@]}
    do
        for layers in ${layersArray[@]}
        do
            #make output dir for paramter settings                                                                                                                         
            echo " -------       new batch run     --------"
            OUTDIR="$maindir/lr_${learningrate}.mtl_${mtl}.layers_${layers}"
            mkdir $OUTDIR
            echo "outdir is " $OUTDIR

            #make subdirs for logging and checkpoints                                                                                                                      
            mkdir "$OUTDIR/log_graph"
            mkdir "$OUTDIR/checkpoints"
            # wrap python call in a string so we can do our fancy redirecting below
            runcmd='python DCNSoundClass.py --outdir $OUTDIR --checkpointing 1 --checkpointPeriod 1000  '
            runcmd+='--numClasses 50 --batchsize 20 --n_epochs 200 --learning_rate ${learningrate}  '
            runcmd+='--keepProb .5 --l1channels 64 --l2channels 32 --fcsize 32 --freqorientation ${orientation}  '
            runcmd+='--numconvlayers ${layers} --adamepsilon ${epsilon} --optimizer ${optimizer} --mtlnumclasses ${mtl}'
			# direct stdout and sterr from each run into their proper directories, but tww so we can still watch
        	eval $runcmd > >(tee $OUTDIR/log.txt) 2> >(tee $OUTDIR.stderr.log >&2)
        done
    done
done

