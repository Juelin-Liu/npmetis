cmake_minimum_required(VERSION 3.16)
project(cppmetis)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)
set(COMPILE_MPI_METIS OFF)

if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/parmetis)
    set(COMPILE_MPI_METIS ON)
endif()

message("Compile MPI METIS: ${COMPILE_MPI_METIS}")

include_directories(third_party/span/include)
add_subdirectory(third_party/cnpy)
include_directories(third_party/cnpy/)
add_subdirectory(cppmetis)
