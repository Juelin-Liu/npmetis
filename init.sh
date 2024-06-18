#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git submodule update --init --recursive

rm -rf ${SCRIPT_DIR}/third_party/build

# build GKlib
pushd ${SCRIPT_DIR}/third_party/GKlib
make config prefix=${SCRIPT_DIR}/third_party/build #gdb=1 debug=1
make -j 
make install 
popd

# build METIS
pushd ${SCRIPT_DIR}/third_party/metis
make config prefix=${SCRIPT_DIR}/third_party/build i64=1 gklib_path=${SCRIPT_DIR}/third_party/build #gdb=1 debug=1
make -j
make install
popd

# # build ParMETIS
# if test -d ${SCRIPT_DIR}/third_party/parmetis/; then
#     pushd ${SCRIPT_DIR}/third_party/parmetis/
#     make config prefix=${SCRIPT_DIR}/third_party/build shared=1 #gdb=1 debug=1
#     make -j
#     make install
#     popd
# fi

# build MT-METIS
pushd ${SCRIPT_DIR}/third_party/mt-metis
./configure --prefix=${SCRIPT_DIR}/third_party/build --edges64bit --weights64bit #--debug
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
# mkdir -p ${SCRIPT_DIR}/third_party/cnpy/build
# pushd ${SCRIPT_DIR}/third_party/cnpy/build
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/third_party/build ..
# cmake --build . -j
# cmake --install .
# popd