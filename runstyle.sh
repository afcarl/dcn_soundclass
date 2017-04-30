#!/bin/bash                                                                                                                                                        
#    nohup ./runstyle.sh >>styleout/2017.04.29/multilog.txt 2>&1 &                                                                                                 
# Individual logs will also still get stored in their respective directories                                                                                       
source activate tflow2
DATE=`date +%Y.%m.%d`
maindir=styleout/$DATE
mkdir $maindir

statefile=testmodel/state.pickle
iter=2000

noise=.2
randArray=(0 1)
contentArray=(BeingRural5.0 agf5.0)
styleArray=(sheepfarm5.0 agf5.0 BeingRural5.0 )

for  rand in ${randArray[@]}
do
    for content in ${contentArray[@]}
    do
        for style in ${styleArray[@]}
        do

           #make output dir for paramter settings                                                                                                                 \
                                                                                                                                                                   
            echo " -------       new batch run     --------"
            OUTDIR="$maindir/content_${content}.style_${style}.rand_${rand}"
            mkdir $OUTDIR
            echo "outdir is " $OUTDIR

            #make subdirs for logging and checkpoints                                                                                                             \
                                                                                                                                                                   
            mkdir "$OUTDIR/log_graph"
            mkdir "$OUTDIR/checkpoints"

                        runcmd='python style_transfer.py --content ${content} --style ${style}  --noise ${noise} --outdir $OUTDIR '
                        runcmd+='--stateFile ${statefile} --iter $iter --alpha 10 --beta 200 --randomize ${rand}'

                        eval $runcmd > >(tee $OUTDIR/log.txt) 2> >(tee $OUTDIR.stderr.log >&2)
        done
    done
done

