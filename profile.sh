#!/usr/bin/env bash
trap 'kill -INT $BGPID' TERM INT

function post() {
  # Postprocessing to produce .svg.
  echo "post processing"
  if [[ $1 = "mem" ]]; then
    latest="$(ls -t ${path_name}/mem.heap.*| head -1)"
    if [[ -n "${latest}" ]]; then
      /home/weiming/.local/bin/pprof --svg ${exec_comm} ${latest} > "${path_name}/mem.svg"
    fi
  elif [[ $1 = "cpu" ]]; then
    rm -v "${path_name}/${name}.svg"
    # pprof --callgrind ${exec_comm} "${path_name}/cpu.prof" > "${path_name}/callgrind.out"
    /home/weiming/.local/bin/pprof --svg ${exec_comm} "${path_name}/cpu.prof" > "${path_name}/${name}.svg"
  fi
}

# name="$1"
# output="${@:2:$#}"
build_type="RelWithDebInfo"
name="leduc_sbcfr_1_${build_type}_$1"
path="./results"
path_name="${path}/${name}"
# mkdir -p ${path_name}
exec_comm="./build/${build_type}/deep_cfr/run_sbcfr1"
if [[ $# = 1 ]]; then
  params="--use_regret_net=true --use_policy_net=true --use_tabular=false --num_gpus=0 \
--num_cpus=1 --actors=1 --memory_size=1000000 --policy_memory_size=10000000 --cfr_batch_size=1000 \
--train_batch_size=4096 --train_steps=16 --policy_train_steps=16 \
--inference_batch_size=100 --inference_threads=1 --inference_cache=100000 \
--omp_threads=4 --exp_evaluation_window=false --game=leduc_poker --cfr_mode=SubSbCFR \
--checkpoint_freq=1000000 --max_steps=10000000 --graph_def=deep_cfr"
  profile_lib="/usr/local/lib/libprofiler.so"
  memcheck_lib="/usr/local/lib/libtcmalloc.so"
  preload_lib=""
  if [[ $1 = "mem" ]]; then
    rm -v "${path_name}/mem.heap*"
    preload_lib="LD_PRELOAD=${memcheck_lib} HEAPPROFILE=${path_name}/mem.heap"
  elif [[ $1 = "cpu" ]]; then
    rm -v "${path_name}/cpu.prof"
    preload_lib="${preload_lib} LD_PRELOAD=${profile_lib} CPUPROFILE=${path_name}/cpu.prof"
  fi
  
  eval "${preload_lib} ${exec_comm} ${params} --path=${path_name}" &
  BGPID=$!
  wait ${BGPID}
  trap - TERM INT
  wait ${BGPID}
  echo "killed ${BGPID}"
  post $@
elif [[ $2 = "post" ]]; then
  # Postprocessing to produce .svg.
  post $@
fi

