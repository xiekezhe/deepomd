#!/usr/bin/env bash

source pre_run.sh

omp_thread=16
actors=128
num_cpus=1

export CUDA_VISIBLE_DEVICES=-1

./build/deep_cfr/run_sbescfr --use_regret_net=true --use_policy_net=true --use_tabular=false --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=40000000 --policy_memory_size=40000000 --cfr_batch_size=10000 \
--train_batch_size=10000 --train_steps=40 --policy_train_steps=4000 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=1000000 \
--omp_threads=$omp_thread --exp_evaluation_window=true --game=FHP_poker --evaluation_window=10 --max_evaluation_window=100 \
--average_type=LinearOpponent --weight_type=Linear \
--checkpoint_freq=1000000 --max_steps=10000000 --graph_def=sbescfr --suffix=$RANDOM --verbose=true --cfr_rm_scale=0.1
