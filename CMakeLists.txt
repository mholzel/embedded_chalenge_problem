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
set(kernels double_everything.cl)
foreach (kernel IN ITEMS ${kernels})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/cl/${kernel} ${CMAKE_CURRENT_BINARY_DIR}/cl/${kernel} COPYONLY)
endforeach ()

# Set up the actual build target
add_executable(embedded_challenge_problem src/main.cpp)

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