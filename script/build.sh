#!/bin/bash
#生成执行文件所在路径
build_dir=../build
#CMakeLists.txt文件所在目录
CMAKELISTS_DIR=../
#添加-p参数设置CMAKE_CURRENT_BINARY_DIR为build_dir
mkdir -p $build_dir
cd $build_dir
cmake -D CMAKE_BUILD_TYPE=Release $CMAKELISTS_DIR
make -j8

