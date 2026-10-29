#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Mathias Dall-Hansen
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 OUTPUT_DIRECTORY" >&2; exit 1; }
[[ $(uname -m) == x86_64 ]] || { echo "This build targets Linux x86_64" >&2; exit 1; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$1"
output=$(cd "$1" && pwd)
cuda="/usr/local/cuda-${CUDA_VER:-13.2}"
trt_include=${TRT_INCLUDE_DIR:-/usr/include/x86_64-linux-gnu}
trt_lib=${TRT_LIB_DIR:-/usr/lib/x86_64-linux-gnu}
eigen_include=${EIGEN_INCLUDE_DIR:-/usr/include/eigen3}

# These paths become Makefile recipe arguments; reject shell/make metacharacters.
for path in "$root" "$output" "$cuda" "$trt_include" "$trt_lib" "$eigen_include"; do
    [[ "$path" =~ ^[/A-Za-z0-9_.-]+$ ]] || { echo "Unsupported build path: $path" >&2; exit 1; }
done
version=$(printf '#include <NvInferVersion.h>\nNV_TENSORRT_MAJOR NV_TENSORRT_MINOR NV_TENSORRT_PATCH NV_TENSORRT_BUILD\n' |
    g++ -E -P -x c++ -I"$trt_include" - | tail -n 1 | tr ' ' '.')
[[ "$version" == 11.* ]] || { echo "TensorRT 11 headers required, got $version" >&2; exit 1; }

bundle="$output/nvinfer-trt11"
[[ ! -e "$bundle" ]] || { echo "Output already exists: $bundle" >&2; exit 1; }
mkdir -p "$bundle/lib"
cflags="-fPIC -std=c++14 -O2 -DNDEBUG -Wall -Werror -Wno-deprecated-declarations -isystem $trt_include -isystem $root/includes -isystem $cuda/include"
libs="-shared -Wl,-no-undefined -L$trt_lib -lnvinfer -lnvinfer_plugin -lnvonnxparser -L$cuda/lib64 -lcudart -L$bundle/lib -lnvds_inferlogger -ldl -Wl,-rpath,'\$\$ORIGIN'"
nvcc="$cuda/bin/nvcc --generate-code=arch=compute_75,code=sm_75 --generate-code=arch=compute_86,code=sm_86 --generate-code=arch=compute_89,code=sm_89"

make -B -C "$root/src/utils/nvdsinfer_logger" CUDA_VER="${CUDA_VER:-13.2}" CFLAGS="$cflags" LIBS=-shared
cp "$root/src/utils/nvdsinfer_logger/libnvds_inferlogger.so" "$bundle/lib/"
make -B -C "$root/src/utils/nvdsinferutils" CUDA_VER="${CUDA_VER:-13.2}" PKGS= \
    CFLAGS="$cflags -fvisibility=hidden -I. -Iinclude -Idbscan -I../nvdsinfer -I../nvdsinfer_logger -isystem $eigen_include" LIBS="$libs -lcrypto"
cp "$root/src/utils/nvdsinferutils/libnvds_inferutils.so" "$bundle/lib/"
make -B -j2 -C "$root/src/utils/nvdsinfer" CUDA_VER="${CUDA_VER:-13.2}" NVCC="$nvcc" \
    CFLAGS="$cflags" LIBS="$libs -lpthread -lnvds_inferutils"
cp "$root/src/utils/nvdsinfer/libnvds_infer.so" "$bundle/lib/"

for library in "$bundle"/lib/*.so; do
    needed=$(readelf -d "$library")
    if grep -E 'lib(nvinfer|nvonnxparser)[^]]*\.so\.10' <<< "$needed"; then
        echo "Unexpected TensorRT 10 dependency: $library" >&2; exit 1
    fi
    dependencies=$(LD_LIBRARY_PATH="$bundle/lib:$trt_lib:$cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$library")
    if grep 'not found' <<< "$dependencies"; then
        echo "Unresolved dependency: $library" >&2; exit 1
    fi
done
readelf -d "$bundle/lib/libnvds_infer.so" | grep -F 'libnvinfer.so.11'

deepstream_version=$(sed -n 's/^NVDS_VERSION:=[[:space:]]*//p' "$root/src/utils/nvdsinfer/Makefile")
[[ "$deepstream_version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || { echo "Cannot identify source SDK version" >&2; exit 1; }
{
    echo "source=$(git -C "$root" rev-parse HEAD)"
    echo "deepstream=$deepstream_version"
    echo "tensorrt=$version"
    echo "cuda=${CUDA_VER:-13.2}"
    echo "gpu_targets=75,86,89"
    echo "validation=build-and-link-only; GPU inference must be tested separately"
    "$cuda/bin/nvcc" --version
} > "$bundle/BUILDINFO"
cp "$root/LICENSE" "$bundle/LICENSE"
cp "$root/THIRD_PARTY_LICENSES.txt" "$bundle/THIRD_PARTY_LICENSES.txt"
(cd "$bundle" && sha256sum lib/*.so BUILDINFO LICENSE THIRD_PARTY_LICENSES.txt > SHA256SUMS)
tar -czf "$output/nvinfer-trt11-linux-x86_64.tar.gz" -C "$output" nvinfer-trt11
