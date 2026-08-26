#pragma once

#include <cmath>
#include <ostream>

/**
 * @file Dual.hh
 * @brief Forward-mode automatic differentiation via dual numbers.
 */

// This header fixes the public interface (templated on the scalar
// coefficient type T) and structure only — every function body marked TODO
// below is a placeholder, so the project compiles and the test suite in
// tests/test_dual.cc runs (and fails "red") before you've implemented
// anything. Fill in the bodies one at a time and watch tests turn green.
//
// See the roadmap notes (Stage 1) for the design decisions behind this
// interface: the variable()/constant() seeding convention, why the compound
// assignment operators should be implemented first, and the subtlety around
// operator order in *= and /=.

namespace autodiff {
namespace forward {

/**
 * @brief Dual number of the form `value + derivative * eps`, with `eps^2 = 0`.
 *
 * Carries a function value alongside its first derivative with respect to
 * a single seeded independent variable; arithmetic and elementary functions
 * propagate the derivative through the chain rule.
 *
 * @tparam T scalar coefficient type (e.g. `float`, `double`).
 */
template <typename T>
class Dual {
 public:
  //! Underlying scalar type.
  using value_type = T;

  /**
   * @brief Builds the zero dual number.
   */
  constexpr Dual() noexcept = default;

  /**
   * @brief Builds a dual number from a plain scalar (derivative = 0).
   *
   * Implicit on purpose: lets plain literals (e.g. `2.0`) participate in
   * expressions with Dual without an explicit cast.
   *
   * @param value primal value.
   */
  constexpr Dual(T value) noexcept : value_(value), derivative_(T(0)) {}

  /**
   * @brief Builds a dual number from primal and derivative parts.
   *
   * @param value primal value.
   * @param derivative derivative coefficient.
   */
  constexpr Dual(T value, T derivative) noexcept
      : value_(value), derivative_(derivative) {}

  /**
   * @brief Seeds an independent variable (derivative = 1).
   *
   * Use for whichever weight you're currently differentiating with respect
   * to.
   *
   * @param value variable value.
   * @return Dual dual number with unit derivative.
   */
  [[nodiscard]] static constexpr Dual variable(T value) noexcept {
    return Dual(value, T(1));
  }

  /**
   * @brief Seeds a value with no dependency on the differentiation variable.
   *
   * Equivalent to the implicit constructor; makes "this is deliberately a
   * constant" explicit at call sites (inputs, other weights, targets).
   *
   * @param value constant value.
   * @return Dual dual number with zero derivative.
   */
  [[nodiscard]] static constexpr Dual constant(T value) noexcept {
    return Dual(value, T(0));
  }

  /// @return T the primal value.
  [[nodiscard]] constexpr T value() const noexcept { return value_; }

  /// @return T the derivative coefficient.
  [[nodiscard]] constexpr T derivative() const noexcept {
    return derivative_;
  }

  /**
   * @brief Adds a dual number in place.
   *
   * @param other addend.
   * @return Dual& reference to the updated object.
   */
  constexpr Dual& operator+=(const Dual& other) noexcept {
    // TODO(you): implement.
    (void)other;
    return *this;
  }

  /**
   * @brief Subtracts a dual number in place.
   *
   * @param other subtrahend.
   * @return Dual& reference to the updated object.
   */
  constexpr Dual& operator-=(const Dual& other) noexcept {
    // TODO(you): implement.
    (void)other;
    return *this;
  }

  /**
   * @brief Multiplies by a dual number in place (product rule).
   *
   * @param other factor.
   * @return Dual& reference to the updated object.
   */
  constexpr Dual& operator*=(const Dual& other) noexcept {
    // TODO(you): implement the product rule. Careful: compute the new
    // derivative_ BEFORE you overwrite value_, or you'll use the updated
    // value in the derivative formula by mistake.
    (void)other;
    return *this;
  }

  /**
   * @brief Divides by a dual number in place (quotient rule).
   *
   * @param other divisor.
   * @return Dual& reference to the updated object.
   */
  constexpr Dual& operator/=(const Dual& other) noexcept {
    // TODO(you): implement the quotient rule. Same ordering caveat as *=.
    (void)other;
    return *this;
  }

  /**
   * @brief Unary negation.
   *
   * @return Dual negated dual number.
   */
  [[nodiscard]] constexpr Dual operator-() const noexcept {
    // TODO(you): implement.
    return *this;
  }

 private:
  T value_ = T(0);
  T derivative_ = T(0);
};

// --- Binary arithmetic operators: built on the compound-assignment ops. ---

template <typename T>
constexpr Dual<T> operator+(Dual<T> a, const Dual<T>& b) noexcept {
  a += b;
  return a;
}

template <typename T>
constexpr Dual<T> operator-(Dual<T> a, const Dual<T>& b) noexcept {
  a -= b;
  return a;
}

template <typename T>
constexpr Dual<T> operator*(Dual<T> a, const Dual<T>& b) noexcept {
  a *= b;
  return a;
}

template <typename T>
constexpr Dual<T> operator/(Dual<T> a, const Dual<T>& b) noexcept {
  a /= b;
  return a;
}

// --- Comparisons: compare the value only — needed for relu's branch, and
// later for Eigen's internals once Dual is used inside Eigen::Matrix. ---

template <typename T>
constexpr bool operator==(const Dual<T>& a, const Dual<T>& b) noexcept {
  // TODO(you): implement.
  (void)a;
  (void)b;
  return false;
}

template <typename T>
constexpr bool operator!=(const Dual<T>& a, const Dual<T>& b) noexcept {
  return !(a == b);
}

template <typename T>
constexpr bool operator<(const Dual<T>& a, const Dual<T>& b) noexcept {
  // TODO(you): implement.
  (void)a;
  (void)b;
  return false;
}

template <typename T>
constexpr bool operator>(const Dual<T>& a, const Dual<T>& b) noexcept {
  return b < a;
}

template <typename T>
constexpr bool operator<=(const Dual<T>& a, const Dual<T>& b) noexcept {
  return !(b < a);
}

template <typename T>
constexpr bool operator>=(const Dual<T>& a, const Dual<T>& b) noexcept {
  return !(a < b);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Dual<T>& x) {
  os << "(" << x.value() << " + " << x.derivative() << "eps)";
  return os;
}

// --- Elementary functions ---
// Free `template <typename T>` functions that pull in the right overload
// via `using std::foo;` (ADL), so the same template works for `float`,
// `double`, or a nested autodiff type. Each TODO follows the same pattern:
// value() = f(x.value()), derivative() = f'(x.value()) * x.derivative()
// (chain rule).

template <typename T>
Dual<T> exp(const Dual<T>& x) {
  // TODO(you): implement. Hint: d/dx exp(x) = exp(x), so you can reuse the
  // computed value in the derivative.
  using std::exp;
  (void)x;
  return Dual<T>();
}

template <typename T>
Dual<T> log(const Dual<T>& x) {
  // TODO(you): implement. d/dx log(x) = 1/x. Undefined for x <= 0.
  using std::log;
  (void)x;
  return Dual<T>();
}

template <typename T>
Dual<T> sqrt(const Dual<T>& x) {
  // TODO(you): implement. d/dx sqrt(x) = 1/(2*sqrt(x)). Undefined for x < 0.
  using std::sqrt;
  (void)x;
  return Dual<T>();
}

/**
 * @brief Raises a Dual to a CONSTANT power (not `Dual^Dual`).
 *
 * @tparam T scalar coefficient type.
 * @param x base.
 * @param p constant exponent.
 * @return Dual result with derivative `p * x^(p-1) * x.derivative()`.
 */
template <typename T>
Dual<T> pow(const Dual<T>& x, T p) {
  // TODO(you): implement. d/dx x^p = p * x^(p-1).
  using std::pow;
  (void)x;
  (void)p;
  return Dual<T>();
}

template <typename T>
Dual<T> sin(const Dual<T>& x) {
  // TODO(you): implement. d/dx sin(x) = cos(x).
  using std::cos;
  using std::sin;
  (void)x;
  return Dual<T>();
}

template <typename T>
Dual<T> cos(const Dual<T>& x) {
  // TODO(you): implement. d/dx cos(x) = -sin(x).
  using std::cos;
  using std::sin;
  (void)x;
  return Dual<T>();
}

template <typename T>
Dual<T> tanh(const Dual<T>& x) {
  // TODO(you): implement. d/dx tanh(x) = 1 - tanh(x)^2 (reuse the value).
  using std::tanh;
  (void)x;
  return Dual<T>();
}

// --- NN-specific activations ---

/**
 * @brief Logistic sigmoid, `1 / (1 + exp(-x))`.
 */
template <typename T>
Dual<T> sigmoid(const Dual<T>& x) {
  // TODO(you): implement directly (don't compose from exp/+// for
  // numerical stability). sigmoid(x) = 1/(1+exp(-x)),
  // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
  using std::exp;
  (void)x;
  return Dual<T>();
}

/**
 * @brief Rectified linear unit.
 */
template <typename T>
Dual<T> relu(const Dual<T>& x) {
  // TODO(you): implement. Pick and DOCUMENT your subgradient convention at
  // exactly x == 0 (0 is the conventional choice) — write it here as a
  // comment once decided.
  (void)x;
  return Dual<T>();
}

}  // namespace forward
}  // namespace autodiff
