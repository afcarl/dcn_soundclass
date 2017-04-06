#!/bin/bash
source activate tflow2
python DCNSoundClass.py  --numClasses 2 --batchsize 8 --n_epochs 2 --learning_rate .001 --keepProb .5
