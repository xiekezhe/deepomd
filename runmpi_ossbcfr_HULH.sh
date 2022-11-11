#!/usr/bin/env bash

source pre_run.sh

./build/deep_cfr/run_ossbcfr --use_regret_net=true --use_policy_net=true --num_gpus=0 \
--num_cpus=1 --actors=100 --memory_size=1000000 --policy_memory_size=10000000 --cfr_batch_size=1000 \
--train_batch_size=128 --train_steps=16 --policy_train_steps=16 \
--inference_batch_size=100 --inference_threads=1 --inference_cache=100000 \
--omp_threads=8 --evaluation_window=100000000 --first_evaluation=100000000 --exp_evaluation_window=true --game=HULH_poker \
--checkpoint_freq=1000000 -sync_period=1 --max_steps=100000000 --graph_def=ossbcfr --suffix=$RANDOM --verbose
