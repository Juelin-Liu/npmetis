cmake_minimum_required(VERSION 3.20)
project(pymetis)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(Python REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD)

add_subdirectory(third_party/pybind11)
# add_subdirectory(pymetis)

find_package(MPI)
if(MPI_FOUND)
    message("MPI found")
    set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
    include_directories(${MPI_INCLUDE_PATH})
    message("MPI_INCLUDE_PATH ${MPI_INCLUDE_PATH}")
    message("MPI_LINK_FLAGS ${MPI_LINK_FLAGS}")
    message("MPI_COMPILE_FLAGS ${MPI_COMPILE_FLAGS}")
    add_subdirectory(pmetis)
endif()