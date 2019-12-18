#!/bin/sh

rm -r paddle
cp /data/jwozna/Paddle/paddle/ . -r
cp/data/jwozna/Paddle/build/paddle/fluid/platform/profiler* paddle/fluid/platform/

cd build

rm -r *
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=/data/jwozna/Paddle/build/fluid_install_dir -DCMAKE_BUILD_TYPE=Release -DUSE_PROFILER=ON ..
make -j30

cd -
