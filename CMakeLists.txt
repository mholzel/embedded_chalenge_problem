cmake_minimum_required(VERSION 3.11)

project(embedded_challenge_problem)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

message("OpenCV_LIBRARIES : ${OpenCV_LIBRARIES}")

add_executable(embedded_challenge_problem src/main.cpp)

target_include_directories(embedded_challenge_problem
        PRIVATE include
        PRIVATE ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(embedded_challenge_problem
        PRIVATE ${OpenCV_LIBS}
        PRIVATE stdc++fs
        PRIVATE Threads::Threads
        )
