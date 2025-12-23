cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(CameraAwareDepth VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Find dependencies
message(STATUS "Finding dependencies...")

# PyTorch C++ (LibTorch)
find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${TORCH_INCLUDE_DIRS}")

# Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# Include FetchContent module for dependency fetching
include(FetchContent)

# nlohmann_json (for JSON parsing)
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, using FetchContent")
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# yaml-cpp (for YAML config parsing)
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found, using FetchContent")
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
    )
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# cxxopts (for command line parsing)
find_package(cxxopts QUIET)
if(NOT cxxopts_FOUND)
    message(STATUS "cxxopts not found, using FetchContent")
    FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.0.0
    )
    FetchContent_MakeAvailable(cxxopts)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/preprocessing
    ${CMAKE_SOURCE_DIR}/src/data
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/layers
    ${CMAKE_SOURCE_DIR}/src/loss
    ${CMAKE_SOURCE_DIR}/src/training
    ${CMAKE_SOURCE_DIR}/src/evaluation
    ${CMAKE_SOURCE_DIR}/src/visualization
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(PREPROCESSING_SOURCES
    src/preprocessing/ray_direction_computer.cpp
)

set(DATA_SOURCES
    src/data/sunrgbd_loader.cpp
)

# Libraries

# Ray Direction Library
add_library(ray_direction_lib STATIC
    ${PREPROCESSING_SOURCES}
)
target_link_libraries(ray_direction_lib
    Eigen3::Eigen
)

# Data Loader Library
add_library(data_loader_lib STATIC
    ${DATA_SOURCES}
)
target_link_libraries(data_loader_lib
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

# Executables

# Data validation tool
add_executable(validate_sunrgbd
    scripts/validate_sunrgbd.cpp
)
target_link_libraries(validate_sunrgbd
    Eigen3::Eigen
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

# Ray direction preprocessing tool
add_executable(preprocess_rays
    src/preprocessing/preprocess_rays_main.cpp
    ${PREPROCESSING_SOURCES}
)
target_link_libraries(preprocess_rays
    ray_direction_lib
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

# Training executable (Week 3)
option(BUILD_TRAINING "Build training executable" ON)
if(BUILD_TRAINING)
    add_executable(train
        src/training/train_main.cpp
    )
    target_link_libraries(train
        ${TORCH_LIBRARIES}
        data_loader_lib
        Eigen3::Eigen
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        /opt/homebrew/lib/libyaml-cpp.dylib
    )
    target_include_directories(train PRIVATE
        ${CMAKE_SOURCE_DIR}
    )
    set_property(TARGET train PROPERTY CXX_STANDARD 17)

    message(STATUS "Building training executable")
endif()

# Evaluation executable (Week 4)
option(BUILD_EVALUATION "Build evaluation executable" OFF)
if(BUILD_EVALUATION)
    add_executable(evaluate
        src/evaluation/evaluate_main.cpp
    )
    target_link_libraries(evaluate
        ${TORCH_LIBRARIES}
        data_loader_lib
        Eigen3::Eigen
        ${OpenCV_LIBS}
        yaml-cpp
        cxxopts
    )
    set_property(TARGET evaluate PROPERTY CXX_STANDARD 17)

    message(STATUS "Building evaluation executable")
endif()

# Test executables
option(BUILD_TESTS "Build test executables" ON)
if(BUILD_TESTS)
    # Data loader test
    add_executable(test_dataloader
        tests/test_dataloader.cpp
    )
    target_link_libraries(test_dataloader
        data_loader_lib
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
    )
    set_property(TARGET test_dataloader PROPERTY CXX_STANDARD 17)

    # Model tests (Week 2)
    add_executable(test_models
        tests/test_models.cpp
    )
    target_link_libraries(test_models
        ${TORCH_LIBRARIES}
        Eigen3::Eigen
    )
    set_property(TARGET test_models PROPERTY CXX_STANDARD 17)
endif()

# Installation
install(TARGETS validate_sunrgbd preprocess_rays train
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/
    DESTINATION include/camera_aware_depth
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Camera-Aware Depth Estimation Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  LibTorch: ${TORCH_VERSION}")
message(STATUS "  Eigen3: ${EIGEN3_VERSION_STRING}")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "")
message(STATUS "Build options:")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "==========================================================")
message(STATUS "")

# Documentation
message(STATUS "To build the project:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "")
message(STATUS "To run data validation:")
message(STATUS "  ./build/validate_sunrgbd ./data/sunrgbd")
message(STATUS "")
message(STATUS "To precompute ray directions:")
message(STATUS "  ./build/preprocess_rays --data_dir ./data/sunrgbd")
message(STATUS "")
