cmake_minimum_required(VERSION 3.15)
project(QuantEngine)

# 开启 C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含头文件路径
include_directories(include)

# 1. 查找 OpenMP
find_package(OpenMP REQUIRED)

# 2. 离线集成 pybind11
add_subdirectory(third_party/pybind11)

# 3. 核心头文件定义
set(CORE_HEADERS
    include/Option.h
    include/BlackScholes.h
    include/MonteCarlo.h
    include/AsianAnalytic.h
    include/Regression.h
)

# 4. 生成 C++ 可执行测试程序 (EXE)
add_executable(QuantEngine src/main.cpp ${CORE_HEADERS})
target_link_libraries(QuantEngine PUBLIC OpenMP::OpenMP_CXX)

# 5. 生成 Python 扩展模块 (PYD)
pybind11_add_module(QuantEngineLib src/bindings.cpp ${CORE_HEADERS})
target_link_libraries(QuantEngineLib PUBLIC OpenMP::OpenMP_CXX)

# 6. 编译器优化设置 (MSVC)
if(MSVC)
    target_compile_options(QuantEngine PRIVATE /O2 /utf-8)
    target_compile_options(QuantEngineLib PRIVATE /O2 /utf-8)
    add_definitions(-D_USE_MATH_DEFINES)
endif()