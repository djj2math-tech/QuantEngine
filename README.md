# High-Performance Option Pricing Engine (C++20 & Pybind11)

[![SJTU Mathematics](https://img.shields.io/badge/SJTU-Mathematics-blue.svg)](https://math.sjtu.edu.cn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance quantitative finance engine for European, Asian, and American option pricing. This project demonstrates a complete workflow from **numerical mathematical modeling** to **low-level C++ optimization** and **Python research integration**.

## 🚀 Core Features
- **C++20 Core**: High-speed Monte Carlo simulations using **OpenMP** for multi-core parallelism.
- **European Options**: Analytic Black-Scholes solution and Monte Carlo with **Antithetic Variates**.
- **Asian Options**: Arithmetic mean pricing via Monte Carlo with **Control Variates** (using Geometric Mean as reference) to achieve 95%+ variance reduction.
- **American Options**: Exercise boundary estimation using the **Longstaff-Schwartz (LSMC)** algorithm with quadratic regression and feature scaling.
- **Hybrid Interface**: High-performance C++ kernels exposed to Python via **Pybind11**.

## 📊 Performance & Convergence (Ablation Study)

### 1. Language Performance: C++ vs NumPy
C++ demonstrates significant advantages in memory locality and fine-grained parallelism.
![Performance Benchmark](docs/images/ultimate_benchmark.png)
*Result: C++ (12T) is significantly faster than vectorized NumPy for $10^7$ simulations.*

### 2. Math Depth: Convergence of Control Variates
Variance reduction techniques allow the engine to achieve higher precision with fewer samples.
![Math Convergence](docs/images/math_convergence.png)
*Result: Control Variates (CV) achieve $10^{-4}$ precision significantly faster than standard MC.*

### 3. HPC Scaling: Parallel Efficiency
Multi-threaded performance scales efficiently across physical cores.
![Scaling Analysis](docs/images/performance_speedup.png)

## 🛠️ Build and Usage
### Prerequisites
- C++20 compatible compiler (MSVC 2019+ or GCC 10+)
- CMake 3.15+
- Python 3.8+ (with `pybind11` for bindings)

### Build Instructions
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
## 📂 Project Structure
include/: Header files (.h) with Doxygen-style comments.
src/: Source files (.cpp) including Pybind11 bindings.
python/: Benchmark and ablation study scripts.
third_party/: Submodules (e.g., pybind11).
## 📝 Author
Your Name
Master of Mathematics, Shanghai Jiao Tong University (SJTU)
```