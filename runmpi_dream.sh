#!/usr/bin/env bash

source pre_run.sh

omp_thread=16
actors=128
num_cpus=1

./build/deep_cfr/run_dream --use_regret_net=true --use_policy_net=true --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=40000000 --policy_memory_size=40000000 \
--cfr_batch_size=50000 --train_batch_size=10000 --train_steps=3000 --policy_train_steps=3000  \
--global_value_memory_size=200000 --global_value_batch_size=512 --global_value_train_steps=1000 \
--omp_threads=$omp_thread --evaluation_window=10 --exp_evaluation_window=true --game=FHP_poker \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=100000 \
--checkpoint_freq=100 --max_steps=10000000 --suffix=$RANDOM --cfr_mode=ESCFR --graph_def=dream
