#!/bin/bash

mkdir build
cd build
cmake ..

make

cp tempo.*.so ../python/tempo
