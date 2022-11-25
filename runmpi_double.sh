#!/usr/bin/env bash

source pre_run.sh

mpirun -n 10 ./build/deep_cfr/run_double_cfr --use_regret_net=true --use_policy_net=true --num_gpus=0 \
--num_cpus=1 --actors=1 --memory_size=40000000 --policy_memory_size=40000000 --cfr_batch_size=10000 \
--train_batch_size=10000 --train_steps=4000 --policy_train_steps=4000 --inference_cache=100000 \
--omp_threads=1 --evaluation_window=10 --exp_evaluation_window=true  --game=FHP_poker \
--checkpoint_freq=100 --suffix=$RANDOM
