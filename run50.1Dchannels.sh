#!/bin/bash
source activate tflow2
python DCNSoundClass1Dchannels.py --checkpointing True --checkpointPeriod 4000  --numClasses 50 --batchsize 20 --n_epochs 400 --learning_rate .001 --keepProb .5 --l1channels 64 --l2channels 48 --fcsize 32
