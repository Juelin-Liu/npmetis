find_package(OpenMP REQUIRED)

pybind11_add_module(pymetis metis_assignment.cc make_sym.cc binding.cc)

target_include_directories(pymetis PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/include)
target_link_directories(pymetis PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/lib)

target_link_libraries(pymetis PRIVATE GKlib metis)
target_link_libraries(pymetis PRIVATE tbb tbbmalloc)
target_link_libraries(pymetis PUBLIC OpenMP::OpenMP_CXX)

pybind11_add_module(pymtmetis mt_metis_assignment.cc make_sym.cc mt_binding.cc)
target_include_directories(pymtmetis PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/include)
target_link_directories(pymtmetis PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/lib)
target_link_libraries(pymtmetis PRIVATE mtmetis)
target_link_libraries(pymtmetis PRIVATE tbb tbbmalloc)
target_link_libraries(pymtmetis PUBLIC OpenMP::OpenMP_CXX)
