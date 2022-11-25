#!/usr/bin/env bash

THIS_DIR=$( cd "$( dirname "$0" )" && pwd )

cd $THIS_DIR
source ./pre_run.sh

$1 --use_regret_net=true \
--use_policy_net=true --num_gpus=0 --num_cpus=1 \
--omp_threads=$2 --game=HULH_poker \
--local_best_response=true --post_evaluation=true --lbr_batch_size=$3 \
--init_strategy_0=$4 \
--init_strategy_1=$5 \
--graph_def=$6 --verbose=true 