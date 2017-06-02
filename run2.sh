#!/bin/bash                                                                                                                                                                
# To store logs and see both stderr and stdout on the screen:  
#    nohup ./run2.sh >>logs/multilog.txt 2>&1 &     
# Individual logs will also still get stored in their respective directories                                                                                                            
source activate tflow2
DATE=`date +%Y.%m.%d`
#maindir=logs.$DATE
#mkdir $maindir

if [ $# -eq 0 ]
  then
    echo "please supply output directory as a command line argument"
    exit
fi

maindir=$1
mkdir $maindir

numconvlayers=2
learningrate=.01
optimizer=adam

orientationArray=(height) #(height)
epsilonArray=(1.0)

mtlArray=(0)

indir=data2

for  mtl in ${mtlArray[@]}
do
    for orientation in ${orientationArray[@]}
    do
        for epsilon in ${epsilonArray[@]}
        do
            #make output dir for paramter settings                                                                                                                         
            echo " -------       new batch run     --------"
            OUTDIR="$maindir/mtl_${mtl}.or_${orientation}.epsilon_${epsilon}"
            mkdir $OUTDIR
            echo "outdir is " $OUTDIR

            #make subdirs for logging and checkpoints                                                                                                                      
            mkdir "$OUTDIR/log_graph"
            mkdir "$OUTDIR/checkpoints"
            # wrap python call in a string so we can do our fancy redirecting below
            runcmd='python DCNSoundClass.py --outdir $OUTDIR --checkpointing 0 --checkpointPeriod 10  --indir ${indir}   '
            runcmd+=' --freqbins 257 --numFrames 856 --convRows 9 '
            runcmd+=' --numClasses 2 --batchsize 20 --n_epochs 100 --learning_rate ${learningrate}  --keepProb .5 '
            runcmd+=' --l1channels 32 --l2channels 32 --fcsize 32 --freqorientation ${orientation}  --learnCondition whenWrong '
            runcmd+=' --adamepsilon ${epsilon} --optimizer ${optimizer} --numconvlayers ${numconvlayers} --mtlnumclasses ${mtl}'
			# direct stdout and sterr from each run into their proper directories, but tww so we can still watch
        	eval $runcmd > >(tee $OUTDIR/log.txt) 2> >(tee $OUTDIR.stderr.log >&2)
        done
    done
done

