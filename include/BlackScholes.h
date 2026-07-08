/**
 * @file BlackScholes.h
 * @brief 欧式期权 Black-Scholes 解析解引擎
 * @details 提供标准的 Black-Scholes 闭式解计算，用于欧式期权估值及作为蒙特卡洛模拟的基准。
 * @author SJTU Mathematics
 */

#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cmath>
#include "Option.h"

 /**
  * @class BlackScholes
  * @brief 提供欧式期权定价的静态数学函数
  */
class BlackScholes {
public:
    /**
     * @brief 标准正态分布累积分布函数 (CDF)
     * @details 采用 std::erfc 实现，在分布尾部具有更好的数值稳定性。
     * @param x 待计算的数值
     * @return 概率值 N(x)
     */
    static double n_cdf(double x) {
        // 使用 1/sqrt(2) 的高精度数值，避免复杂的库依赖以确保编译稳定性
        return 0.5 * std::erfc(-x * 0.70710678118654752440);
    }

    /**
     * @brief 计算欧式期权的理论价格
     * @details 基于资产收益率服从对数正态分布的假设：
     * Call = S * N(d1) - K * exp(-rT) * N(d2)
     * Put  = K * exp(-rT) * N(-d2) - S * N(-d1)
     *
     * @param p 包含 S, K, r, sigma, T 和 type 的期权参数结构体
     * @return 期权的理论公平价值
     */
    static double price(const OptionParams& p) {
        // 边界保护：若到期时间为0，返回内在价值
        if (p.T <= 0.0) {
            if (p.type == OptionType::Call)
                return std::max(p.S - p.K, 0.0);
            else
                return std::max(p.K - p.S, 0.0);
        }

        const double sqrtT = std::sqrt(p.T);
        const double d1 = (std::log(p.S / p.K) + (p.r + 0.5 * p.sigma * p.sigma) * p.T) / (p.sigma * sqrtT);
        const double d2 = d1 - p.sigma * sqrtT;

        if (p.type == OptionType::Call) {
            return p.S * n_cdf(d1) - p.K * std::exp(-p.r * p.T) * n_cdf(d2);
        }
        else {
            return p.K * std::exp(-p.r * p.T) * n_cdf(-d2) - p.S * n_cdf(-d1);
        }
    }
};