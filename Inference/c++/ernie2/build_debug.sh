#!/bin/sh

build_dir=build_debug
build_type=Debug

export PYTHONPATH=/data/wojtuss/repos/PaddlePaddle/Paddle/${build_dir}/python:/data/wojtuss/repos/PaddlePaddle/Paddle/${build_dir}/paddle

rm -r paddle
cp /data/wojtuss/repos/PaddlePaddle/Paddle/paddle/ . -r
cp /data/wojtuss/repos/PaddlePaddle/Paddle/${build_dir}/paddle/fluid/platform/profiler* paddle/fluid/platform/

cd ${build_dir} && rm -r * 
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=/data/wojtuss/repos/PaddlePaddle/Paddle/${build_dir}/fluid_install_dir -DCMAKE_BUILD_TYPE=${build_type} -DUSE_PROFILER=ON .. 
make -j10

cd -
