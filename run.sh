#!/bin/bash
python DCNSoundClass.py --numClasses 50 --batchsize 40 --n_epochs 300 --learning_rate .001 2>&1 | tee log_graph/log.txt
