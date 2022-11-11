#!/usr/bin/env bash

source pre_run.sh

omp_thread=8
actors=1
num_cpus=1

./build/deep_cfr/placeholder \
--use_tabular=false \
--cfr_batch_size=1 \
--cfr_mode=PostSbCFR \
--average_type=Opponent \
--weight_type=Linear \
--omp_threads=$omp_thread \
--max_steps=10010 \
--suffix=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 12)
