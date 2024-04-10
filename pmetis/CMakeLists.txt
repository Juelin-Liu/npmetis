
add_executable(pmetis main.cc utils.cc partition.cc)

target_include_directories(pmetis PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/include)
target_link_directories(pmetis PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/lib)

target_link_libraries(pmetis PRIVATE ${MPI_C_LIBRARY})
target_link_libraries(pmetis PRIVATE parmetis metis)
target_link_libraries(pmetis PRIVATE cnpy)
target_link_libraries(pmetis PRIVATE tbb tbbmalloc)
target_link_libraries(pmetis PRIVATE pybind11::module)
