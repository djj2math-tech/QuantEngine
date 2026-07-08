/**
 * @file Regression.h
 * @brief 最小二乘线性回归工具类
 * @details 专门用于求解二次多项式拟合，是美式期权 Longstaff-Schwartz (LSMC) 算法的核心组件。
 * @author SJTU Mathematics
 */

#pragma once
#include <vector>
#include <cmath>

 /**
  * @class Regression
  * @brief 提供数值回归分析的数学工具
  */
class Regression {
public:
    /**
     * @brief 求解二次多项式拟合 y = a + b*x + c*x^2
     * @details 通过构造正态方程 (Normal Equations) 并利用克拉默法则 (Cramer's Rule) 求解 3x3 线性方程组。
     *          该方法在 LSMC 算法中用于估计当前资产价格与未来现金流期望之间的函数关系。
     *
     * @param x 自变量向量 (在 LSMC 中通常为 ITM 路径的缩放股价)
     * @param y 因变量向量 (在 LSMC 中通常为折现后的未来现金流)
     * @return std::vector<double> 返回回归系数序列 [a, b, c]
     */
    static std::vector<double> solve_quadratic(const std::vector<double>& x, const std::vector<double>& y) {
        size_t n = x.size();

        // s0...s4 存储自变量 x 的各阶矩 (Moments)
        double s0 = (double)n, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        // b0...b2 存储 x 与 y 的交叉矩
        double b0 = 0, b1 = 0, b2 = 0;

        // 构造正态方程组所需的各项系数
        for (size_t i = 0; i < n; ++i) {
            double xi = x[i];
            double xi2 = xi * xi;
            s1 += xi;
            s2 += xi2;
            s3 += xi2 * xi;
            s4 += xi2 * xi2;

            b0 += y[i];
            b1 += xi * y[i];
            b2 += xi2 * y[i];
        }

        // 计算 3x3 系数矩阵的行列式 (Determinant)
        double det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2);

        // 数值稳定性检查：若矩阵近乎奇异 (Singular)，返回零向量以防止除零错误
        if (std::abs(det) < 1e-15) return { 0.0, 0.0, 0.0 };

        // 利用克拉默法则求解系数向量 [a, b, c]
        std::vector<double> res(3);
        res[0] = (b0 * (s2 * s4 - s3 * s3) - s1 * (b1 * s4 - b2 * s3) + s2 * (b1 * s3 - b2 * s2)) / det;
        res[1] = (s0 * (b1 * s4 - b2 * s3) - b0 * (s1 * s4 - s2 * s3) + s2 * (s1 * b2 - b1 * s2)) / det;
        res[2] = (s0 * (s2 * b2 - b1 * s3) - s1 * (s1 * b2 - b0 * s3) + b0 * (s1 * s3 - s2 * s2)) / det;

        return res;
    }
};