#!/usr/bin/env bash

THIS_DIR=$( cd "$( dirname "$0" )" && pwd )

cd $THIS_DIR
source ./pre_run.sh

omp_thread=8
num_cpus=1

./build/deep_cfr/run_deep_cfr --use_regret_net=true \
--use_policy_net=true --num_gpus=0 --num_cpus=$num_cpus \
--omp_threads=$omp_thread --game=leduc_poker \
--play=true  --host=$1 --port=$2 \
--init_strategy_0=./results/05_01_2021.14_29_07.11636/checkpoint-leduc_poker_deep_cfr_policy_0_cpu750 \
--init_strategy_1=./results/05_01_2021.14_29_07.11636/checkpoint-leduc_poker_deep_cfr_policy_0_cpu750 \
--graph_def=deep_cfr --inference_cache=1000000 --verbose=true 