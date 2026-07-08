/**
 * @file Option.h
 * @brief Defines basic option types, parameters, and payoff structures.
 * @author SJTU Mathematics
 */

#pragma once
#include <algorithm>
#include <vector>

 /**
  * @enum OptionType
  * @brief Simple enumeration for Option types: Call or Put.
  */
enum class OptionType {
    Call,
    Put
};

/**
 * @struct OptionParams
 * @brief Container for standard Black-Scholes model parameters.
 */
struct OptionParams {
    double S;          ///< Current spot price of the underlying asset
    double K;          ///< Strike price of the option
    double r;          ///< Risk-free interest rate (annualized)
    double sigma;      ///< Volatility of the underlying asset (annualized)
    double T;          ///< Time to maturity (in years)
    OptionType type;   ///< Option contract type (Call/Put)
};

/**
 * @class Payoff
 * @brief Abstract base class for option payoff calculations.
 *
 * Provides a polymorphic interface for different exercise styles.
 */
class Payoff {
public:
    virtual ~Payoff() {}

    /**
     * @brief Calculates the payoff based on a single terminal price.
     * @param spot The price of the underlying asset at exercise.
     * @return The resulting payoff value.
     */
    virtual double operator()(double spot) const = 0;
};

/**
 * @class PayoffEuropean
 * @brief Implementation of standard European option payoff.
 *
 * Payoff = max(S - K, 0) for Call, max(K - S, 0) for Put.
 */
class PayoffEuropean : public Payoff {
private:
    double K;
    OptionType type;

public:
    PayoffEuropean(double K_, OptionType type_) : K(K_), type(type_) {}

    virtual double operator()(double spot) const override {
        if (type == OptionType::Call)
            return std::max(spot - K, 0.0);
        return std::max(K - spot, 0.0);
    }
};

/**
 * @class PayoffAsian
 * @brief Implementation of Asian option payoff (Path-dependent).
 *
 * The payoff depends on the average price of the asset over a specific period.
 */
class PayoffAsian : public Payoff {
private:
    double K;
    OptionType type;

public:
    PayoffAsian(double K_, OptionType type_) : K(K_), type(type_) {}

    /**
     * @brief Implementation of the base interface (Terminal spot).
     * @note For Asian options, terminal spot alone is insufficient;
     * use calculate_payoff with a full path.
     */
    virtual double operator()(double spot) const override {
        return 0.0;
    }

    /**
     * @brief Calculates payoff based on the arithmetic average of a price path.
     * @param path A vector containing price observations along the asset's trajectory.
     * @return Payoff based on the average price.
     */
    double calculate_payoff(const std::vector<double>& path) const {
        if (path.empty()) return 0.0;

        double sum = 0.0;
        for (double s : path) sum += s;
        double average = sum / static_cast<double>(path.size());

        if (type == OptionType::Call)
            return std::max(average - K, 0.0);
        return std::max(K - average, 0.0);
    }
};