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

MODEL_DIR=/data/jwozna/data/ernie_int8_fc_reshape_transpose
DATA_FILE=/data/jwozna/data/1.8w.bs1
LABEL_FILE=/data/jwozna/data/label.xnli.dev
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

output_prediction=true
if [ $# -ge 6 ]; then
  output_prediction=$6
fi

if [ $# -ge 7 ]; then
  LABEL_FILE=$7
fi


./build/inference --logtostderr \
    --model_dir=${MODEL_DIR} \
    --data=${DATA_FILE} \
    --label=${LABEL_FILE} \
    --repeat=${REPEAT} \
    --use_gpu=${USE_GPU} \
    --num_threads=${num_threads} \
    --profile=${profile} \
    --output_prediction=${output_prediction} \
    --use_int8 \
    #  --remove_scale \
    # --debug \
    # --warmup_steps=1 \
    # --use_analysis=true \
