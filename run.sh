#!/usr/bin/env bash

source pre_run.sh

omp_thread=8
actors=100

./build/deep_cfr/run_deep_cfr \
--use_regret_net=true --use_policy_net=true --use_tabular=true \
--num_gpus=0  --num_cpus=1 --actors=$actors \
--memory_size=10000 --policy_memory_size=400000 --cfr_batch_size=100 \
--train_batch_size=256 --train_steps=100 --policy_train_steps=100 \
--evaluation_window=10 --exp_evaluation_window=true \
--checkpoint_freq=100 --checkpoint_second=10 --max_steps=10000 \
--inference_threads=1 --inference_cache=100000  --inference_batch_size=$actors \
--omp_threads=$omp_thread --game=leduc_poker --graph_def=deep_cfr  \
--verbose=true --suffix=$RANDOM  $* \
