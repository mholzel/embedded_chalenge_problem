cmake_minimum_required(VERSION 3.11)

project(embedded_challenge_problem)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Copy all kernels to the binary location
# Note: at the moment this is not used. We force the path in C++ by specifying relative to the cpp file
file(GLOB kernels cl/*.cl)
foreach (kernel IN ITEMS ${kernels})
    get_filename_component(kern ${kernel} NAME)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/cl/${kern} ${CMAKE_CURRENT_BINARY_DIR}/cl/${kern} COPYONLY)
    #    configure_file(${CMAKE_CURRENT_LIST_DIR}/cl/${kernel} ${CMAKE_CURRENT_BINARY_DIR}/cl/${kernel}.cpp COPYONLY)
    #    set_source_files_properties(cl/consistency_check.cl PROPERTIES LANGUAGE CXX)
endforeach ()

# Set up the actual build target
add_executable(embedded_challenge_problem src/main.cpp
        #        ${CMAKE_CURRENT_BINARY_DIR}/cl/consistency_check.cl.cpp
        )

target_include_directories(embedded_challenge_problem
        PRIVATE include
        PRIVATE ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(embedded_challenge_problem
        PRIVATE OpenCL::OpenCL
        PRIVATE ${OpenCV_LIBS}
        PRIVATE stdc++fs
        PRIVATE Threads::Threads
        )