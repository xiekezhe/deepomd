#!/usr/bin/env bash

source pre_run.sh

omp_thread=8
actors=1
num_cpus=1

./build/deep_cfr/run_sbcfr1 --use_regret_net=true --use_policy_net=true --use_tabular=true --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=400000 --policy_memory_size=400000 --cfr_batch_size=100 \
--train_batch_size=256 --train_steps=16 --policy_train_steps=64 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=100000 \
--omp_threads=$omp_thread --exp_evaluation_window=false --game=leduc_poker \
--checkpoint_freq=100000 --max_steps=100000 --graph_def=deep_cfr --suffix=$RANDOM --verbose=false \
--cfr_rm_amp=1.1 --cfr_rm_damp=0.9 --cfr_rm_scale=1e-5 --cfr_mode=SubSbCFR
