#!/bin/bash

cd /home/tz58/gcn-over-pruned-trees
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/gcn-500 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/gcn-600 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/gcn-700 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/gcn-800 --dataset test
