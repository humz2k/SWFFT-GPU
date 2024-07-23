#!/usr/bin/env bash

clang-format -i src/**/*.cpp
clang-format -i src/**/*.cu
clang-format -i src/**/*.hpp
clang-format -i test/*.hpp
clang-format -i test/*.cpp
clang-format -i examples/*.cpp
clang-format -i include/*.hpp
clang-format -i include/**/*.hpp
clang-format -i include/*.h