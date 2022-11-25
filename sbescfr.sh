#!/usr/bin/env bash

source pre_run.sh

omp_thread=4
actors=1
num_cpus=1

./build/deep_cfr/run_sbescfr --use_regret_net=true --use_policy_net=false --use_tabular=true --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=1000000 --policy_memory_size=10000000 --cfr_batch_size=100 \
--train_batch_size=128 --train_steps=2 --policy_train_steps=2 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=1000000 \
--omp_threads=$omp_thread --exp_evaluation_window=false --game=leduc_poker --evaluation_window=10 \
--average_type=LinearOpponent --weight_type=Constant \
--checkpoint_freq=1000000 --max_steps=10000000 --graph_def=sbescfr --suffix=$RANDOM --verbose=false --cfr_rm_scale=10 --cfr_rm_amp=1.1 --cfr_rm_damp=0.9