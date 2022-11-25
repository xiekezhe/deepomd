#!/usr/bin/env bash

THIS_DIR=$( cd "$( dirname "$0" )" && pwd )

cd $THIS_DIR
source ./pre_run.sh

omp_thread=8
num_cpus=1

./build/deep_cfr/run_deep_cfr --use_regret_net=true \
--use_policy_net=true --num_gpus=0 --num_cpus=$num_cpus \
--omp_threads=$omp_thread --game=HULH_poker \
--play=true  --host=$1 --port=$2 \
--init_strategy_0=./models/checkpoint-HULH_poker_deep_cfr_policy_0_gpu240 \
--init_strategy_1=./models/checkpoint-HULH_poker_deep_cfr_policy_0_gpu240 \
--graph_def=deep_cfr --inference_cache=1000000 --verbose=true 