# Build main 
add_executable(main main.cc utils.cc partition.cc)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/include)
target_link_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/lib)

target_link_libraries(main PRIVATE metis GKlib)
target_link_libraries(main PRIVATE cnpy_mmap)
target_link_libraries(main PRIVATE tbb tbbmalloc)

# Build mt_main 
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(mt_main mt_main.cc utils.cc mt_partition.cc)
    target_include_directories(mt_main PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/include)
    target_link_directories(mt_main PRIVATE ${CMAKE_SOURCE_DIR}/third_party/build/lib)

    target_link_libraries(mt_main PRIVATE mtmetis metis GKlib)
    target_link_libraries(mt_main PRIVATE cnpy_mmap)
    target_link_libraries(mt_main PRIVATE tbb tbbmalloc)
    target_link_libraries(mt_main PUBLIC OpenMP::OpenMP_CXX)
endif()

# Build mpi_main 
 find_package(MPI)
 if(MPI_FOUND)
     set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
     message("MPI found")
     message("MPI_INCLUDE_PATH ${MPI_INCLUDE_PATH}")
     message("MPI_LINK_FLAGS ${MPI_LINK_FLAGS}")
     message("MPI_COMPILE_FLAGS ${MPI_COMPILE_FLAGS}")
     message("MPI_LIBRARIES ${MPI_LIBRARIES}")

     add_executable(mpi_main mpi_main.cc utils.cc mpi_partition.cc)
     target_include_directories(mpi_main PUBLIC ${CMAKE_SOURCE_DIR}/third_party/build/include)
     target_link_directories(mpi_main PUBLIC ${CMAKE_SOURCE_DIR}/third_party/build/lib)
     target_include_directories(mpi_main PUBLIC ${MPI_INCLUDE_PATH})
     target_link_libraries(mpi_main PUBLIC ${MPI_LIBRARIES} )

     target_link_libraries(mpi_main PUBLIC parmetis metis GKlib)
     target_link_libraries(mpi_main PUBLIC cnpy_mmap)
     target_link_libraries(mpi_main PUBLIC tbb tbbmalloc)
 endif()
