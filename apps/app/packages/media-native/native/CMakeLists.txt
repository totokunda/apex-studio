cmake_minimum_required(VERSION 3.22)
project(media_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(media_core STATIC
  src/core/placeholder.cpp
)

target_include_directories(media_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
