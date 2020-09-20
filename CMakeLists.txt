cmake_minimum_required(VERSION 3.10)
project(cloud_registration)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-O2")
endif (CMAKE_BUILD_TYPE)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

include_directories(thirdparty/Sophus)

add_executable(cloud_registration registrate_ply.cpp)
target_link_libraries(cloud_registration ${CERES_LIBRARIES} ${PCL_LIBRARIES})