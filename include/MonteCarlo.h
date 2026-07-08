/**
 * @file MonteCarlo.h
 * @brief 高性能蒙特卡洛定价引擎
 * @details 包含支持多线程加速的欧式、亚式（算术平均）及美式期权定价算法。
 *          集成了对偶变量法 (Antithetic Variates) 与控制变量法 (Control Variates) 等方差缩减技术。
 * @author SJTU Mathematics
 */

#pragma once
#include "Option.h"
#include "AsianAnalytic.h"
#include "Regression.h"
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>

 /**
  * @class MonteCarloEngine
  * @brief 蒙特卡洛模拟核心引擎
  */
class MonteCarloEngine {
public:
    /**
     * @brief 动态设置 OpenMP 并行线程数
     * @param num_threads 线程数量，若为 -1 则使用系统默认设置
     */
    static void set_threads(int num_threads) {
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
    }

    /**
     * @brief 欧式期权定价 (对偶变量法)
     * @details 通过生成互为相反数的随机数路径 Z 和 -Z，利用对偶变量减少模拟方差。
     * @param p 期权基本参数
     * @param payoff 收益计算逻辑
     * @param num_sims 总模拟次数 (建议为偶数)
     * @param num_threads 线程数
     * @return double 期权现值
     */
    static double calculate(const OptionParams& p, const Payoff& payoff, long long num_sims, int num_threads = -1) {
        set_threads(num_threads);
        double total_payoff = 0.0;
        double drift = (p.r - 0.5 * p.sigma * p.sigma) * p.T;
        double vol_sqrt_T = p.sigma * std::sqrt(p.T);
        double df = std::exp(-p.r * p.T);

#pragma omp parallel reduction(+:total_payoff)
        {
            // 为每个线程分配独立的随机数种子，确保并行安全性
            std::mt19937 generator(12345 + omp_get_thread_num());
            std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
            for (long long i = 0; i < num_sims / 2; ++i) {
                double Z = dist(generator);
                // 对偶路径模拟
                total_payoff += payoff(p.S * std::exp(drift + vol_sqrt_T * Z));
                total_payoff += payoff(p.S * std::exp(drift + vol_sqrt_T * (-Z)));
            }
        }
        return df * (total_payoff / num_sims);
    }

    /**
     * @brief 亚式期权定价 - 标准蒙特卡洛 (用于消融实验)
     * @details 模拟离散时间路径，计算算术平均值收益。
     * @param steps 时间采样步数 (如 252 代表每日采样)
     */
    static double calculate_asian_standard(const OptionParams& p, int steps, long long num_sims, int num_threads = -1) {
        set_threads(num_threads);
        double dt = p.T / steps;
        double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
        double vol_sqrt_dt = p.sigma * std::sqrt(dt);
        double total_arith = 0.0;

#pragma omp parallel reduction(+:total_arith)
        {
            std::mt19937 gen(54321 + omp_get_thread_num());
            std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
            for (long long i = 0; i < num_sims; ++i) {
                double s = p.S, sum_a = 0.0;
                for (int j = 0; j < steps; ++j) {
                    s *= std::exp(drift + vol_sqrt_dt * dist(gen));
                    sum_a += s;
                }
                double avg_arith = sum_a / steps;
                total_arith += (p.type == OptionType::Call) ? std::max(avg_arith - p.K, 0.0) : std::max(p.K - avg_arith, 0.0);
            }
        }
        return std::exp(-p.r * p.T) * (total_arith / num_sims);
    }

    /**
     * @brief 亚式期权定价 - 控制变量法 (Variance Reduction)
     * @details 利用几何平均亚式期权的解析解作为控制变量，大幅降低算术平均亚式期权的模拟方差。
     *          数学原理：Arithmetic_Price = MC_Arith - (MC_Geom - Analytic_Geom)
     */
    static double calculate_asian_cv(const OptionParams& p, int steps, long long num_sims, int num_threads = -1) {
        set_threads(num_threads);
        double dt = p.T / steps;
        double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
        double vol_sqrt_dt = p.sigma * std::sqrt(dt);
        double total_arith = 0.0, total_geom = 0.0;

#pragma omp parallel reduction(+:total_arith, total_geom)
        {
            std::mt19937 gen(54321 + omp_get_thread_num());
            std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
            for (long long i = 0; i < num_sims; ++i) {
                double s = p.S, sum_a = 0.0, sum_l = 0.0;
                for (int j = 0; j < steps; ++j) {
                    s *= std::exp(drift + vol_sqrt_dt * dist(gen));
                    sum_a += s;
                    sum_l += std::log(s);
                }
                double avg_arith = sum_a / steps;
                double avg_geom = std::exp(sum_l / steps);

                if (p.type == OptionType::Call) {
                    total_arith += std::max(avg_arith - p.K, 0.0);
                    total_geom += std::max(avg_geom - p.K, 0.0);
                }
                else {
                    total_arith += std::max(p.K - avg_arith, 0.0);
                    total_geom += std::max(p.K - avg_geom, 0.0);
                }
            }
        }
        double df = std::exp(-p.r * p.T);
        double mc_arith = df * (total_arith / num_sims);
        double mc_geom = df * (total_geom / num_sims);
        double analytic_geom = AsianAnalytic::geometric_asian_price(p, steps);
        // 应用控制变量修正
        return mc_arith - (mc_geom - analytic_geom);
    }

    /**
     * @brief 美式期权定价 - Longstaff-Schwartz (LSMC) 算法
     * @details 通过最小二乘回归估计条件期望（延续价值），解决美式期权的最优停止时刻问题。
     * @param steps 时间步数（即行权观察点）
     * @param num_sims 路径模拟总数
     */
    static double calculate_american(const OptionParams& p, int steps, long long num_sims, int num_threads = -1) {
        set_threads(num_threads);
        double dt = p.T / steps;
        double df = std::exp(-p.r * dt);
        double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
        double vol_sqrt_dt = p.sigma * std::sqrt(dt);

        // 阶段 1: 生成并存储所有价格路径
        std::vector<std::vector<double>> paths(num_sims, std::vector<double>(steps + 1));

#pragma omp parallel
        {
            std::mt19937 gen(888 + omp_get_thread_num());
            std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
            for (long long i = 0; i < num_sims; ++i) {
                paths[i][0] = p.S;
                for (int j = 1; j <= steps; ++j) {
                    paths[i][j] = paths[i][j - 1] * std::exp(drift + vol_sqrt_dt * dist(gen));
                }
            }
        }

        // 阶段 2: 终点收益初始化
        std::vector<double> cash_flows(num_sims);
        for (long long i = 0; i < num_sims; ++i) {
            cash_flows[i] = (p.type == OptionType::Call) ? std::max(paths[i][steps] - p.K, 0.0) : std::max(p.K - paths[i][steps], 0.0);
        }

        // 阶段 3: 动态规划倒推 (Backward Induction)
        for (int t = steps - 1; t >= 1; --t) {
            std::vector<double> X, Y;
            std::vector<int> itm_idx;
            for (int i = 0; i < (int)num_sims; ++i) {
                cash_flows[i] *= df; // 折现至当前时间点
                double exercise = (p.type == OptionType::Call) ? std::max(paths[i][t] - p.K, 0.0) : std::max(p.K - paths[i][t], 0.0);
                // 仅对价内 (ITM) 路径进行回归拟合
                if (exercise > 0) {
                    X.push_back(paths[i][t] / p.S); // 特征缩放增强数值稳定性
                    Y.push_back(cash_flows[i]);
                    itm_idx.push_back(i);
                }
            }

            if (itm_idx.size() > 10) {
                // 最小二乘回归拟合延续价值 (Continuation Value)
                std::vector<double> beta = Regression::solve_quadratic(X, Y);
                for (size_t j = 0; j < itm_idx.size(); ++j) {
                    int idx = itm_idx[j];
                    double continuation = beta[0] + beta[1] * X[j] + beta[2] * X[j] * X[j];
                    double exercise = (p.type == OptionType::Call) ? std::max(paths[idx][t] - p.K, 0.0) : std::max(p.K - paths[idx][t], 0.0);
                    // 最优行权判定
                    if (exercise > continuation) cash_flows[idx] = exercise;
                }
            }
        }
        double total = 0;
        for (double c : cash_flows) total += c;
        return (total / num_sims) * df;
    }
};