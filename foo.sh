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
learningrate=.01
orientation=channels
layers=2
mtl=8
l2channels=64
fcsize=32

l1channelsArray=(16 32)
epsilonArray=(0.1 1.0)


for l1channels in ${l1channelsArray[@]}
do
    for l2channels in ${l2channelsArray[@]}
    do
        for fcsize in ${fcsizeArray[@]}
        do
            #make output dir for paramter settings                                                                                                                               
            echo " -------       new batch run     --------"
            OUTDIR="$maindir/l1r_${l1channels}.l2_${l2channels}.fc_${fcsize}"
            mkdir $OUTDIR
            echo "outdir is " $OUTDIR

            #keep a copy of this run file                                                                                                                                        
            me=`basename "$0"`
            cp $me $OUTDIR

            #make subdirs for logging and checkpoints                                                                                                                            
            mkdir "$OUTDIR/log_graph"
            mkdir "$OUTDIR/checkpoints"
            mkdir "$OUTDIR/stderr"
            # wrap python call in a string so we can do our fancy redirecting below                                                                                              
            runcmd='python DCNSoundClass.py --outdir $OUTDIR --checkpointing 1 --checkpointPeriod 1000  '
            runcmd+='--numClasses 50 --batchsize 20 --n_epochs 200 --learning_rate ${learningrate}  '
            runcmd+='--keepProb .5 --l1channels ${l1channels} --l2channels ${l2channels} --fcsize ${fcsize} --freqorientation ${orientation}  '
            runcmd+='--numconvlayers ${layers} --adamepsilon ${epsilon} --optimizer ${optimizer} --mtlnumclasses ${mtl}'
                        # direct stdout and sterr from each run into their proper directories, but tww so we can still watch                                                     
                eval $runcmd > >(tee $OUTDIR/log.txt) 2> >(tee $OUTDIR/stderr/stderr.log >&2)
        done
    done
done


