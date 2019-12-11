#!/bin/bash

set -xe

gpu_id="-1"
if [ $# -ge 1 ]; then
  gpu_id="$1"
fi

num_threads=1
if [ $# -ge 2 ]; then
  num_threads=$2
fi

if [ ${gpu_id} -eq "-1" ]; then
  USE_GPU=false
  export CUDA_VISIBLE_DEVICES=""
else
  USE_GPU=true
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi

# MODEL_DIR=/data/wojtuss/models/ernie_quant_int8/
# MODEL_DIR=/data/wojtuss/models/transformed_qat_int8_model/
MODEL_DIR=/data/wojtuss/models/origin/
DATA_FILE=/data/wojtuss/datasets/Ernie/2-inputs/1.8w.bs1
REPEAT=1

if [ $# -ge 3 ]; then
  MODEL_DIR=$3
fi

if [ $# -ge 4 ]; then
  DATA_FILE=$4
fi

profile=1
if [ $# -ge 5 ]; then
  profile=$5
fi

output_prediction=false
if [ $# -ge 6 ]; then
  output_prediction=$6
fi

# build_dir=build_release
# build_dir=build_release_develop
build_dir=build_release_3d-fc_1.1

# INT8
# ./${build_dir}/inference --logtostderr \
    # --model_dir=${MODEL_DIR} \
    # --data=${DATA_FILE} \
    # --repeat=${REPEAT} \
    # --use_gpu=${USE_GPU} \
    # --num_threads=${num_threads} \
    # --profile=${profile} \
    # --output_prediction=${output_prediction} \
    # --use_int8 \
    # --squash \
    # # --short \
    # # --enable_memory_optim \
    # # --remove_scale \

    # # --debug \
    # # --warmup_steps=1 \
    # # --use_analysis=true \

# FP32
./${build_dir}/inference --logtostderr \
    --model_dir=${MODEL_DIR} \
    --data=${DATA_FILE} \
    --repeat=${REPEAT} \
    --use_gpu=${USE_GPU} \
    --num_threads=${num_threads} \
    --profile=${profile} \
    --output_prediction=${output_prediction} \
    --enable_memory_optim \
    # --use_int8 \
    # --squash \
    # --short \
    # --remove_scale \

    # --debug \
    # --warmup_steps=1 \
    # --use_analysis=true \
