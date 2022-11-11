#!/usr/bin/env bash

source pre_run.sh

omp_thread=8
actors=1
num_cpus=1

./build/deep_cfr/run_sbcfr --use_regret_net=true --use_policy_net=true --use_tabular=true --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=1000000 --policy_memory_size=10000000 --cfr_batch_size=1000 \
--train_batch_size=128 --train_steps=16 --policy_train_steps=16 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=100000 \
--omp_threads=$omp_thread --exp_evaluation_window=false --game=leduc_poker --cfr_mode=SubSbCFR \
--checkpoint_freq=1000000 --max_steps=10000000 --graph_def=ossbcfr --suffix=$RANDOM --verbose=false
