find_package(OpenMP REQUIRED)

pybind11_add_module(pymetis mt_metis_assignment.cc make_sym.cc binding.cc)

target_include_directories(pymetis PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/include)
target_link_directories(pymetis PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/lib)
target_link_libraries(pymetis PRIVATE mtmetis)
target_link_libraries(pymetis PRIVATE cnpy_mmap)
target_link_libraries(pymetis PRIVATE tbb tbbmalloc)
target_link_libraries(pymetis PUBLIC OpenMP::OpenMP_CXX)