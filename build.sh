#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# build GKlib
rm -rf ${SCRIPT_DIR}/third_party/build
pushd ${SCRIPT_DIR}/third_party/parmetis/metis/GKlib
make config prefix=${SCRIPT_DIR}/third_party/build openmp=set
make -j 
make install 
popd

# build METIS
pushd ${SCRIPT_DIR}/third_party/parmetis/metis
make config prefix=${SCRIPT_DIR}/third_party/build
make -j
make install
popd

# build ParMETIS
pushd ${SCRIPT_DIR}/third_party/parmetis/
make config prefix=${SCRIPT_DIR}/third_party/build shared=1
make -j
make install
popd

# build oneTBB
mkdir -p ${SCRIPT_DIR}/third_party/oneTBB/build
pushd ${SCRIPT_DIR}/third_party/oneTBB/build 
cmake -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/third_party/build .. 
cmake --build . -j 
cmake --install . 
popd

# build cnpy
mkdir -p ${SCRIPT_DIR}/third_party/cnpy/build
pushd ${SCRIPT_DIR}/third_party/cnpy/build 
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/third_party/build .. 
cmake --build . -j
cmake --install .
popd

# rm -rf build
# cmake -B build -GNinja && cmake --build build -j