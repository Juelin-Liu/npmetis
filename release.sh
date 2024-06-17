#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cmake -B ${SCRIPT_DIR}/build -DCMAKE_BUILD_TYPE=Release && cmake --build ${SCRIPT_DIR}/build -j