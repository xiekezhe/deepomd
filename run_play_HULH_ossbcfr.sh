#!/usr/bin/env bash

THIS_DIR=$( cd "$( dirname "$0" )" && pwd )

cd $THIS_DIR
source ./pre_run.sh

omp_thread=8
num_cpus=1

./build/deep_cfr/run_ossbcfr --use_regret_net=true \
--use_policy_net=true --num_gpus=0 --num_cpus=$num_cpus \
--omp_threads=$omp_thread --game=HULH_poker \
--play=true  --host=$1 --port=$2 \
--init_strategy_0=./models/checkpoint-HULH_poker_ossbcfr_policy_0_gpu83000 \
--init_strategy_1=./models/checkpoint-HULH_poker_ossbcfr_policy_1_gpu83000 \
--graph_def=ossbcfr --inference_cache=1000000 --verbose=true 