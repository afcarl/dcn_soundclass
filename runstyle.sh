#!/bin/bash                                                                                                                                                        
<<<<<<< HEAD
#    nohup ./runstyle.sh >>styleout/2017.05.02/multilog.txt 2>&1 &                                                                                                 
=======
#    nohup ./runstyle.sh >>styleout/2017.04.29/multilog.txt 2>&1 &                                                                                                 
>>>>>>> b89fbe6be843caa0b4bcf1957e0fe2a48fb0fa2b
# Individual logs will also still get stored in their respective directories                                                                                       
source activate tflow2
DATE=`date +%Y.%m.%d`
maindir=styleout/$DATE
mkdir $maindir

<<<<<<< HEAD
statefile=testmodel/1d/state.pickle
iter=4000
beta=200
noise=.4
=======
statefile=testmodel/state.pickle
iter=2000

noise=.2
>>>>>>> b89fbe6be843caa0b4bcf1957e0fe2a48fb0fa2b
randArray=(0 1)
contentArray=(BeingRural5.0 agf5.0 Superstylin5.0 roosters5.0 Nancarrow5.0 Toys5.0 inc5.0 sheepfarm5.0)
styleArray=(BeingRural5.0 agf5.0 Superstylin5.0 roosters5.0 Nancarrow5.0 Toys5.0 inc5.0 sheepfarm5.0)

for  rand in ${randArray[@]}
do
    for content in ${contentArray[@]}
    do
        for style in ${styleArray[@]}
        do

           #make output dir for paramter settings                                                                                                                 \
                                                                                                                                                                   
            echo " -------       new batch run     --------"
<<<<<<< HEAD
            OUTDIR="$maindir/content_${content}.style_${style}.beta_${beta}.rand_${rand}"
=======
            OUTDIR="$maindir/content_${content}.style_${style}.rand_${rand}"
>>>>>>> b89fbe6be843caa0b4bcf1957e0fe2a48fb0fa2b
            mkdir $OUTDIR
            echo "outdir is " $OUTDIR

            #make subdirs for logging and checkpoints                                                                                                             \
                                                                                                                                                                   
            mkdir "$OUTDIR/log_graph"
            mkdir "$OUTDIR/checkpoints"

                        runcmd='python style_transfer.py --content ${content} --style ${style}  --noise ${noise} --outdir $OUTDIR '
<<<<<<< HEAD
                        runcmd+='--stateFile ${statefile} --iter $iter --alpha 10 --beta ${beta} --randomize ${rand}'
=======
                        runcmd+='--stateFile ${statefile} --iter $iter --alpha 10 --beta 200 --randomize ${rand}'
>>>>>>> b89fbe6be843caa0b4bcf1957e0fe2a48fb0fa2b

                        eval $runcmd > >(tee $OUTDIR/log.txt) 2> >(tee $OUTDIR.stderr.log >&2)
        done
    done
done

