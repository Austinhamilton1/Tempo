cmake_minimum_required(VERSION 3.16)

project(Tempo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- pybind11 ---
add_subdirectory(extern/pybind11)

# --- Include directories ---
include_directories(cpp/include)

# --- C++ Core Library ---
add_library(tempo_core
    cpp/src/event.cc
    cpp/src/tgraph.cc
)

target_include_directories(tempo_core PUBLIC cpp/include)

# --- Python Module ---
pybind11_add_module(tempo
    python/bindings.cpp
)

target_link_libraries(tempo PRIVATE tempo_core)