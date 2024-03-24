#!/bin/bash

pip uninstall -y HROCH

# install current release from pypi
# pip install HROCH==1.4.10

# build and install stable branch from sources
rm -rf tmp
mkdir tmp
cd tmp
# clone and build c++ sources
# TODO: add commit hash when final version
git clone -b classification_test --depth 1 --single-branch https://github.com/janoPig/sr_core.git
cd sr_core/Hroch
echo build hroch binary..
clang++-18 *.cpp -o hroch.bin -DNDEBUG -fveclib=libmvec -std=c++20 -O3 -mavx2 -Wall -Wextra -fno-math-errno -fno-signed-zeros -funsafe-math-optimizations -ftree-vectorize -fno-exceptions -shared -fPIC
echo done!

cd ../..
# clone and install python package
git clone -b classification_test --depth 1  --single-branch https://github.com/janoPig/HROCH.git

cp -f -v ./sr_core/Hroch/hroch.bin ./HROCH/HROCH/hroch.bin
cd HROCH
pip install .
cd ../..
rm -rf tmp
