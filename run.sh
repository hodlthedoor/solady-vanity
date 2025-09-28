#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

BUILD_DIR="build"
BINARY_NAME="create3_cuda"
OUTPUT_PATH="$BUILD_DIR/$BINARY_NAME"

mkdir -p "$BUILD_DIR"

nvcc -O3 -Xcompiler -fno-exceptions -arch=sm_89 -o "$OUTPUT_PATH" src/main.cu

"$OUTPUT_PATH" "$@"
