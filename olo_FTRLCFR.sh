#!/usr/bin/env bash

source pre_run.sh

omp_thread=8
actors=1
num_cpus=1

./build/deep_cfr/run_online_learning \
--use_tabular=false \
--cfr_batch_size=1 \
--cfr_mode=FTRLCFR \
--average_type=Opponent \
--omp_threads=$omp_thread \
--game=leduc18_poker \
--cfr_rm_scale=1 \
--suffix=$RANDOM
