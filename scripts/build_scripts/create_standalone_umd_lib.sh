#!/bin/bash

# This script will create a standalone UMD artifact with as many static
# libraries as possible for downstream use. This should ideally be in CI
# for UMD.

set -euo pipefail

cd tt_metal/third_party/umd
git clean -xffd
cmake -B build -G Ninja -DTT_UMD_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --target umd_tests

cd build/test/umd

# Remove hardcoded RPATH
find . -type f | xargs -n 1 -I {} patchelf --remove-rpath {}

mkdir -p device
mkdir -p lib
mkdir -p tests

cp ../../lib/libdevice.so lib/
cp -r ../../_deps/nanomsg-build/*.so* lib/
cp -r ../../_deps/libuv-build/*.so* lib/
cp -r ../../../device/bin/ device/
cp -r ../../../tests/soc_descs/ tests/

cd ../../../
mv tt_metal/third_party/umd/build/test/umd _umd
