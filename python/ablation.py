"""
Ablation Study and Performance Benchmarking Script
Project: High-Performance Quant Pricing Engine (C++20 Core)
Description: This script evaluates the computational efficiency and mathematical 
             convergence of the C++ engine compared to Python/NumPy baselines.
Author: SJTU Mathematics
"""

import QuantEngineLib as qe
import time
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Global Configuration & Baseline Setup
# ==============================================================================
# Standard test parameters: S=100, K=100, r=5%, sigma=20%, T=1yr
p = qe.OptionParams(100.0, 100.0, 0.05, 0.2, 1.0, qe.OptionType.Call)

def price_european_numpy(params, sims):
    """
    Standard NumPy vectorized implementation for European Option Pricing.
    Used as a baseline to demonstrate C++ memory efficiency and raw speed.
    """
    drift = (params.r - 0.5 * params.sigma**2) * params.T
    vol_sqrt_t = params.sigma * np.sqrt(params.T)
    
    # Vectorized path generation (Draws from N(0,1))
    z = np.random.standard_normal(sims)
    st = params.S * np.exp(drift + vol_sqrt_t * z)
    
    # Payoff calculation
    if params.type == qe.OptionType.Call:
        payoffs = np.maximum(st - params.K, 0)
    else:
        payoffs = np.maximum(params.K - st, 0)
        
    return np.exp(-params.r * params.T) * np.mean(payoffs)

# ==============================================================================
# 2. Experiment A: Language & Parallelism Benchmark
# ==============================================================================
def run_language_benchmark():
    """
    Compares execution time across different technologies:
    NumPy (Baseline) vs C++ Single-Thread vs C++ Multi-Thread (OpenMP).
    """
    print("\n" + "="*60)
    print("Experiment A: Language Performance Comparison (NumPy vs C++v20)")
    print("="*60)
    
    sim_counts = [10**5, 5*10**5, 10**6, 5*10**6, 10**7]
    times_numpy = []
    times_cpp_1t = []
    times_cpp_12t = []

    for sc in sim_counts:
        # NumPy Baseline
        start = time.time()
        price_european_numpy(p, sc)
        times_numpy.append(time.time() - start)
        
        # C++ Core (Single-Threaded)
        start = time.time()
        qe.price_european_mc(p, sc, 1)
        times_cpp_1t.append(time.time() - start)
        
        # C++ Core (Multi-Threaded - OpenMP)
        start = time.time()
        qe.price_european_mc(p, sc, 12)
        times_cpp_12t.append(time.time() - start)
        
        print(f"Sims: {sc:10d} | NumPy: {times_numpy[-1]:.4f}s | C++(1T): {times_cpp_1t[-1]:.4f}s | C++(12T): {times_cpp_12t[-1]:.4f}s")

    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(sim_counts, times_numpy, 'r-s', label='Python (NumPy Vectorized)')
    plt.plot(sim_counts, times_cpp_1t, 'g-^', label='C++ (Single-Thread)')
    plt.plot(sim_counts, times_cpp_12t, 'b-o', label='C++ (12-Threads OpenMP)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Benchmarking: Execution Time vs Simulation Count", fontsize=14)
    plt.xlabel("Number of Simulations (Log Scale)", fontsize=12)
    plt.ylabel("Time (Seconds, Log Scale)", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("ultimate_benchmark.png")
    print(">> Plot saved: ultimate_benchmark.png")

# ==============================================================================
# 3. Experiment B: Multi-threading Scaling Study
# ==============================================================================
def run_performance_study():
    """
    Evaluates the scaling efficiency of the OpenMP implementation.
    Tests how the speedup factor evolves with increasing thread counts.
    """
    print("\n" + "="*60)
    print("Experiment B: High-Performance Computing (HPC) Scaling Study")
    print("="*60)
    sims = 10**7
    thread_counts = [1, 2, 4, 8, 12]
    times = []

    for tc in thread_counts:
        start = time.time()
        qe.price_european_mc(p, sims, tc)
        duration = time.time() - start
        times.append(duration)
        print(f"Threads: {tc:2d} | Time: {duration:.4f}s | Speedup: {times[0]/duration:.2f}x")

    plt.figure(figsize=(10, 5), dpi=100)
    speedup = [times[0] / t for t in times]
    plt.plot(thread_counts, speedup, 'o-', linewidth=2, label='Measured Speedup')
    plt.plot(thread_counts, thread_counts, '--', color='gray', label='Ideal Speedup (Linear)')
    plt.title("Scaling Analysis: Parallel Efficiency", fontsize=14)
    plt.xlabel("Number of Threads", fontsize=12)
    plt.ylabel("Speedup Factor", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("performance_speedup.png")
    print(">> Plot saved: performance_speedup.png")

# ==============================================================================
# 4. Experiment C: Variance Reduction & Convergence
# ==============================================================================
def run_math_convergence_study():
    """
    Mathematical Ablation: Standard Monte Carlo vs. Control Variate (CV) Method.
    Demonstrates how mathematical optimization reduces estimation variance.
    """
    print("\n" + "="*60)
    print("Experiment C: Convergence Study (Standard vs Control Variates)")
    print("="*60)
    
    # Generate high-precision reference value (True Price) using 5M sims with CV
    true_price = qe.price_asian_cv(p, 252, 5 * 10**6, 12)
    
    sim_counts = [1000, 5000, 10000, 50000, 100000, 500000]
    errors_standard = []
    errors_cv = []

    for sc in sim_counts:
        res_std = qe.price_asian_standard(p, 252, sc, 12)
        res_cv = qe.price_asian_cv(p, 252, sc, 12)
        errors_standard.append(abs(res_std - true_price))
        errors_cv.append(abs(res_cv - true_price))
        print(f"Sims: {sc:7d} | Std Error: {errors_standard[-1]:.6f} | CV Error: {errors_cv[-1]:.6f}")

    plt.figure(figsize=(10, 5), dpi=100)
    # Log-Log plot is standard for showing O(N^-0.5) convergence rates
    plt.loglog(sim_counts, errors_standard, 'r-o', label='Standard Asian MC')
    plt.loglog(sim_counts, errors_cv, 'g-s', label='Control Variate Asian MC')
    plt.title("Convergence Depth: Impact of Variance Reduction", fontsize=14)
    plt.xlabel("Number of Simulations (Log Scale)", fontsize=12)
    plt.ylabel("Absolute Pricing Error (Log Scale)", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("math_convergence.png")
    print(">> Plot saved: math_convergence.png")

# ==============================================================================
# 5. Main Execution Flow
# ==============================================================================
if __name__ == "__main__":
    start_all = time.time()
    
    run_language_benchmark()      # Performance Tier Study
    run_performance_study()       # Parallel Hardware Scaling
    run_math_convergence_study()  # Mathematical Efficiency Study
    
    print(f"\n[Success] Ablation studies completed in {time.time() - start_all:.2f} seconds.")