#!/bin/bash

cd /home/tz58/gcn-over-pruned-trees
#for seed in "100" "200" "300" "400"; do
for seed in "100"; do
python /home/tz58/gcn-over-pruned-trees/train.py \
    --id sgr2-${seed} \
    --seed ${seed} \
    --prune_k 1 \
    --lr 0.3 \
    --no-rnn \
    --num_epoch 100 \
    --pooling max \
    --mlp_layers 2 \
    --pooling_l2 0.003 \
    --num_layers 2 \
    --graph_model SGR \
    --gcn_dropout 0.5;
done
