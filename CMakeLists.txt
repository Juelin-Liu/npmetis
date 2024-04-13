cmake_minimum_required(VERSION 3.16)
project(cppmetis)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)

include_directories(third_party/span/include)
add_subdirectory(cppmetis)