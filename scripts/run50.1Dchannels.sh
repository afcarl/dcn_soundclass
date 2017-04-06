#!/bin/bash
source activate tflow2
python DCNSoundClass1Dchannels.py --checkpointing True --checkpointPeriod 1000  --numClasses 50 --batchsize 20 --n_epochs 200 --learning_rate .01 --keepProb .5
