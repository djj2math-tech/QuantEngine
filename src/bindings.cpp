/**
 * @file bindings.cpp
 * @brief Python 绑定接口实现
 * @details 利用 pybind11 将 C++ 高性能定价引擎封装为 Python 模块。
 *          实现了从底层 C++ 算力到 Python 研究环境的无缝对接。
 * @author SJTU Mathematics
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Option.h"
#include "BlackScholes.h"
#include "MonteCarlo.h"

namespace py = pybind11;

/**
 * @brief 定义 Python 模块 QuantEngineLib
 */
PYBIND11_MODULE(QuantEngineLib, m) {
    m.doc() = "High Performance Quant Pricing Engine (C++20 Core) - Developed by SJTU Math Student";

    // --- 导出枚举：OptionType ---
    py::enum_<OptionType>(m, "OptionType", "Enum for Call and Put options")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put)
        .export_values();

    // --- 导出结构体：OptionParams ---
    py::class_<OptionParams>(m, "OptionParams", "Container for Black-Scholes model parameters")
        .def(py::init<double, double, double, double, double, OptionType>(),
            py::arg("S"), py::arg("K"), py::arg("r"), py::arg("sigma"), py::arg("T"), py::arg("type"),
            "Constructor for OptionParams")
        .def_readwrite("S", &OptionParams::S, "Current asset price")
        .def_readwrite("K", &OptionParams::K, "Strike price")
        .def_readwrite("r", &OptionParams::r, "Risk-free interest rate")
        .def_readwrite("sigma", &OptionParams::sigma, "Asset volatility")
        .def_readwrite("T", &OptionParams::T, "Time to maturity (years)")
        .def_readwrite("type", &OptionParams::type, "Option type (Call/Put)");

    // --- 接口 1: 欧式期权 Black-Scholes 解析解 ---
    m.def("price_european_bs", &BlackScholes::price,
        "Calculate European option price using the analytical Black-Scholes formula.",
        py::arg("p"));

    // --- 接口 2: 欧式期权 蒙特卡洛模拟 (对偶变量法) ---
    m.def("price_european_mc", [](const OptionParams& p, long long sims, int threads) {
        PayoffEuropean payoff(p.K, p.type);
        return MonteCarloEngine::calculate(p, payoff, sims, threads);
        },
        "Calculate European option price using Monte Carlo with Antithetic Variates.",
        py::arg("p"), py::arg("sims"), py::arg("threads") = -1);

    // --- 接口 3: 亚式期权 标准蒙特卡洛 (算术平均) ---
    m.def("price_asian_standard", &MonteCarloEngine::calculate_asian_standard,
        "Calculate Asian option price (Arithmetic Mean) using standard Monte Carlo simulation.",
        py::arg("p"), py::arg("steps"), py::arg("sims"), py::arg("threads") = -1);

    // --- 接口 4: 亚式期权 控制变量蒙特卡洛 (数学深度体现) ---
    m.def("price_asian_cv", &MonteCarloEngine::calculate_asian_cv,
        "Calculate Asian option price using Control Variates (Geometric Mean) to significantly reduce variance.",
        py::arg("p"), py::arg("steps"), py::arg("sims"), py::arg("threads") = -1);

    // --- 接口 5: 美式期权 LSMC 算法 (Longstaff-Schwartz) ---
    m.def("price_american_mc", &MonteCarloEngine::calculate_american,
        "Calculate American option price using the Longstaff-Schwartz (LSMC) algorithm with quadratic regression.",
        py::arg("p"), py::arg("steps"), py::arg("sims"), py::arg("threads") = -1);
}