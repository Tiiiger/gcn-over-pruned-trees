#!/bin/bash

cd /home/tz58/gcn-over-pruned-trees

python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/csgr2-nodropout-200 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/csgr2-nodropout-300 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/csgr2-nodropout-400 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/csgr2-nodropout-500 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/csgr2-nodropout-600 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/csgr2-nodropout-700 --dataset test
python /home/tz58/gcn-over-pruned-trees/eval.py saved_models/csgr2-nodropout-800 --dataset test
