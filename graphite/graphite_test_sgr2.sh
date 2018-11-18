#!/bin/bash

cd /home/tz58/gcn-over-pruned-trees
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/sgr2-100 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/sgr2-200 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/sgr2-300 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/sgr2-400 --dataset test
