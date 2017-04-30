#!/bin/bash  
#    nohup ./runstyle.sh >>styleout/2017.04.29/multilog.txt 2>&1 &     
# Individual logs will also still get stored in their respective directories 
source activate tflow2

statefile=testmodel/state.pickle
iter=600

noise=.2
rand=0
content=BeingRural5.0
style=agf5.0

python style_transfer.py --content ${content} --style ${style}  --noise ${noise} --outdir testout  --stateFile ${statefile} --iter $iter --alpha 10 --beta 200 --randomize ${rand}
