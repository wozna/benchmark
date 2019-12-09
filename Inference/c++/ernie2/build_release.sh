#!/bin/sh

build_dir=build_release
build_type=Release

export PYTHONPATH=/home/wojtuss/repos/PaddlePaddle/Paddle/${build_dir}/python:/home/wojtuss/repos/PaddlePaddle/Paddle/${build_dir}/paddle

rm -r paddle
cp /home/wojtuss/repos/PaddlePaddle/Paddle/paddle/ . -r
cp /home/wojtuss/repos/PaddlePaddle/Paddle/${build_dir}/paddle/fluid/platform/profiler* paddle/fluid/platform/

cd ${build_dir} && rm -r *
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=/home/wojtuss/repos/PaddlePaddle/Paddle/${build_dir}/fluid_install_dir -DCMAKE_BUILD_TYPE=${build_type} -DUSE_PROFILER=ON ..
make -j14

cd -
