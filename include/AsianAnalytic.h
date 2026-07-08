/**
 * @file AsianAnalytic.h
 * @brief 几何平均亚式期权解析解计算
 * @details 提供离散监测下几何平均亚式期权的闭式解（Closed-form Solution）。
 *          由于对数正态分布随机变量的几何平均值仍服从对数正态分布，因此可以通过修正参数
 *          调用类似 Black-Scholes 的公式进行精确定价。
 * @author SJTU Mathematics
 */

#pragma once
#include "Option.h"
#include "BlackScholes.h"

 /**
  * @class AsianAnalytic
  * @brief 提供几何平均亚式期权的解析定价功能
  */
class AsianAnalytic {
public:
    /**
     * @brief 计算几何平均亚式期权的解析解 (用于控制变量法)
     * @details 根据离散监测步数 (Steps) 对波动率和漂移项进行调整。
     *          该结果作为算术平均亚式期权蒙特卡洛模拟的“基准”，用于显著降低方差。
     *
     * @param p 期权基本参数结构体
     * @param steps 路径模拟的时间步数（离散监测频率）
     * @return double 几何平均亚式期权的公平价格
     */
    static double geometric_asian_price(const OptionParams& p, int steps) {
        // 1. 针对离散监测进行波动率修正 (Volatility Adjustment)
        // 修正公式考虑了离散观测点的相关性
        double sigma_adj = p.sigma * std::sqrt((2.0 * steps + 1.0) / (6.0 * (steps + 1.0)));

        // 2. 修正漂移项 (Drift Adjustment)
        // 确保几何平均路径在风险中性测度下的期望演化正确
        double mu_adj = 0.5 * (p.r - 0.5 * p.sigma * p.sigma) + 0.5 * sigma_adj * sigma_adj;

        // 3. 使用修正后的参数构造类似 Black-Scholes 的定价结构
        // 计算调整后的 d1 和 d2
        double d1 = (std::log(p.S / p.K) + (mu_adj + 0.5 * sigma_adj * sigma_adj) * p.T) / (sigma_adj * std::sqrt(p.T));
        double d2 = d1 - sigma_adj * std::sqrt(p.T);

        // 4. 折现并返回理论价格
        double price = std::exp(-p.r * p.T) * (p.S * std::exp(mu_adj * p.T) * BlackScholes::n_cdf(d1) - p.K * BlackScholes::n_cdf(d2));

        return price;
    }
};