#!/usr/bin/env bash

source pre_run.sh

omp_thread=8
actors=1
num_cpus=1

./build/deep_cfr/run_raw \
--use_tabular=false \
--cfr_batch_size=1 \
--cfr_mode=PSbCFR \
--average_type=Opponent \
--weight_type=Linear \
--omp_threads=$omp_thread \
--game=leduc_poker \
--cfr_rm_scale=1e-6 \
--suffix=$RANDOM
