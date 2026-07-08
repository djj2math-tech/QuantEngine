/**
 * @file main.cpp
 * @brief 定价引擎主程序：算法验证与性能基准测试
 * @details 该程序用于演示和验证 C++ 核心引擎的功能，包括：
 *          1. 欧式期权解析解与蒙特卡洛解的精度对比。
 *          2. 亚式期权在普通模拟与控制变量优化下的消融实验。
 *          3. 美式期权 LSMC 算法的定价展示。
 * @author SJTU Mathematics
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include "Option.h"
#include "BlackScholes.h"
#include "MonteCarlo.h"
#include "AsianAnalytic.h"

 /**
  * @class Timer
  * @brief 高精度计时器，用于性能分析 (Benchmarking)
  */
class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    Timer() { start_time = std::chrono::high_resolution_clock::now(); }

    /**
     * @brief 计算自对象创建以来流逝的时间（秒）
     * @return double 耗时秒数
     */
    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }
};

int main() {
    // ---------------------------------------------------------
    // 基础参数设定 (S=100, K=100, r=5%, sigma=20%, T=1yr)
    // ---------------------------------------------------------
    OptionParams call_params{ 100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call };
    OptionParams put_params{ 100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Put };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "====================================================" << std::endl;
    std::cout << "   SJTU Quant Engine - Performance Benchmark" << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << "Threads Used : " << omp_get_max_threads() << std::endl;

    // ---------------------------------------------------------
    // [Step 1] 欧式期权验证 (European Option)
    // 验证蒙特卡洛模拟对 Black-Scholes 解析解的收敛性
    // ---------------------------------------------------------
    std::cout << "\n[Step 1: European Option]" << std::endl;
    long long euro_sims = 10000000;
    double bs_euro = BlackScholes::price(call_params);

    Timer t_euro;
    PayoffEuropean payoff_euro(call_params.K, call_params.type);
    double mc_euro = MonteCarloEngine::calculate(call_params, payoff_euro, euro_sims);

    std::cout << "BS Analytic Price : " << bs_euro << std::endl;
    std::cout << "MC Estimate Price : " << mc_euro << std::endl;
    std::cout << "Execution Time    : " << t_euro.elapsed() << " s" << std::endl;

    // ---------------------------------------------------------
    // [Step 2] 亚式期权消融实验 (Asian Option Ablation Study)
    // 对比普通蒙特卡洛与引入控制变量法 (Control Variates) 后的性能与精度
    // ---------------------------------------------------------
    std::cout << "\n[Step 2: Asian Option Ablation Study]" << std::endl;
    long long asian_sims = 1000000;
    int asian_steps = 252;

    // A. 普通蒙特卡洛模拟 (Standard MC)
    Timer t_std;
    double price_std = MonteCarloEngine::calculate_asian_standard(call_params, asian_steps, asian_sims);
    double time_std = t_std.elapsed();

    // B. 带控制变量优化的模拟 (Control Variate MC)
    Timer t_cv;
    double price_cv = MonteCarloEngine::calculate_asian_cv(call_params, asian_steps, asian_sims);
    double time_cv = t_cv.elapsed();

    double geom_analytic = AsianAnalytic::geometric_asian_price(call_params, asian_steps);

    std::cout << "Standard Asian MC  : " << price_std << " (Time: " << time_std << "s)" << std::endl;
    std::cout << "CV Asian MC        : " << price_cv << " (Time: " << time_cv << "s)" << std::endl;
    std::cout << "Geom Analytic Ref  : " << geom_analytic << " (Internal CV Reference)" << std::endl;
    std::cout << "Difference (Std-CV): " << std::abs(price_std - price_cv) << std::endl;

    // ---------------------------------------------------------
    // [Step 3] 美式期权定价 (American Put - LSMC)
    // 验证 Longstaff-Schwartz 算法处理最优停止问题的能力
    // ---------------------------------------------------------
    std::cout << "\n[Step 3: American Put (LSMC)]" << std::endl;
    long long am_sims = 100000;
    int am_steps = 50;

    Timer t_am;
    double am_price = MonteCarloEngine::calculate_american(put_params, am_steps, am_sims);

    std::cout << "European Put (BS)   : " << BlackScholes::price(put_params) << std::endl;
    std::cout << "American Put (LSMC) : " << am_price << std::endl;
    std::cout << "Execution Time      : " << t_am.elapsed() << " s" << std::endl;

    std::cout << "\n====================================================" << std::endl;
    std::cout << "All Pricing Tasks Completed." << std::endl;
    std::cout << "Press Enter to exit...";
    std::cin.get();
    return 0;
}