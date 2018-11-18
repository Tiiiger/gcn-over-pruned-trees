#!/bin/bash

cd /home/tz58/gcn-over-pruned-trees
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/cgcn-100 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/cgcn-200 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/cgcn-300 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/cgcn-400 --dataset test
