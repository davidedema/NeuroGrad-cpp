#pragma once

#include <cmath>
#include <ostream>

// Forward-mode automatic differentiation scalar type.
//
// This header fixes the public interface only — every function body below
// is a placeholder (marked TODO) so the project compiles and the test suite
// in tests/test_dual.cc runs (and fails "red") before you've implemented
// anything. Fill in the bodies one at a time and watch tests turn green.
//
// See the roadmap notes (Stage 1) for the design decisions behind this
// interface: the variable()/constant() seeding convention, why the compound
// assignment operators should be implemented first, and the subtlety around
// operator order in *= and /=.

namespace autodiff {
namespace forward {

class Dual {
 public:
  Dual() : value_(0.0), derivative_(0.0) {}

  // Implicit on purpose: lets plain double literals participate in
  // expressions with Dual without an explicit cast (e.g. `x * 2.0`).
  Dual(double value) : value_(value), derivative_(0.0) {}

  Dual(double value, double derivative)
      : value_(value), derivative_(derivative) {}

  // Seed a variable you're differentiating with respect to (derivative = 1).
  static Dual variable(double value) { return Dual(value, 1.0); }

  // A value with no dependency on the seeded variable (derivative = 0).
  // Equivalent to the implicit constructor, but makes "this is deliberately
  // a constant" explicit at call sites.
  static Dual constant(double value) { return Dual(value, 0.0); }

  double value() const { return value_; }
  double derivative() const { return derivative_; }

  // --- Compound assignment: implement these first; the binary operators
  // further down can (and should) be written in terms of them. ---
  Dual& operator+=(const Dual& other) {
    // TODO(you): implement.
    (void)other;
    return *this;
  }

  Dual& operator-=(const Dual& other) {
    // TODO(you): implement.
    (void)other;
    return *this;
  }

  Dual& operator*=(const Dual& other) {
    // TODO(you): implement the product rule. Careful: compute the new
    // derivative_ BEFORE you overwrite value_, or you'll use the updated
    // value in the derivative formula by mistake.
    (void)other;
    return *this;
  }

  Dual& operator/=(const Dual& other) {
    // TODO(you): implement the quotient rule. Same ordering caveat as *=.
    (void)other;
    return *this;
  }

  Dual operator-() const {
    // TODO(you): implement (unary negation).
    return *this;
  }

 private:
  double value_;
  double derivative_;
};

// --- Binary arithmetic operators ---
inline Dual operator+(Dual a, const Dual& b) { a += b; return a; }
inline Dual operator-(Dual a, const Dual& b) { a -= b; return a; }
inline Dual operator*(Dual a, const Dual& b) { a *= b; return a; }
inline Dual operator/(Dual a, const Dual& b) { a /= b; return a; }

// --- Comparisons: compare value_ only (needed for e.g. relu's branch, and
// later for Eigen's internals once Dual is used inside Eigen::Matrix). ---
inline bool operator==(const Dual& a, const Dual& b) {
  // TODO(you): implement.
  (void)a; (void)b;
  return false;
}
inline bool operator!=(const Dual& a, const Dual& b) { return !(a == b); }

inline bool operator<(const Dual& a, const Dual& b) {
  // TODO(you): implement.
  (void)a; (void)b;
  return false;
}
inline bool operator>(const Dual& a, const Dual& b) { return b < a; }
inline bool operator<=(const Dual& a, const Dual& b) { return !(b < a); }
inline bool operator>=(const Dual& a, const Dual& b) { return !(a < b); }

inline std::ostream& operator<<(std::ostream& os, const Dual& x) {
  os << "(" << x.value() << " + " << x.derivative() << "eps)";
  return os;
}

// --- Elementary functions ---
// Each TODO follows the same pattern: value_ = f(x.value()), and
// derivative_ = f'(x.value()) * x.derivative()  (chain rule).

inline Dual exp(const Dual& x) {
  // TODO(you): implement. Hint: d/dx exp(x) = exp(x), so you can reuse the
  // computed value in the derivative.
  (void)x;
  return Dual();
}

inline Dual log(const Dual& x) {
  // TODO(you): implement. d/dx log(x) = 1/x. Undefined for x <= 0.
  (void)x;
  return Dual();
}

inline Dual sqrt(const Dual& x) {
  // TODO(you): implement. d/dx sqrt(x) = 1/(2*sqrt(x)). Undefined for x < 0.
  (void)x;
  return Dual();
}

inline Dual pow(const Dual& x, double p) {
  // TODO(you): implement for a CONSTANT exponent p (not Dual^Dual).
  // d/dx x^p = p * x^(p-1).
  (void)x; (void)p;
  return Dual();
}

inline Dual sin(const Dual& x) {
  // TODO(you): implement. d/dx sin(x) = cos(x).
  (void)x;
  return Dual();
}

inline Dual cos(const Dual& x) {
  // TODO(you): implement. d/dx cos(x) = -sin(x).
  (void)x;
  return Dual();
}

inline Dual tanh(const Dual& x) {
  // TODO(you): implement. d/dx tanh(x) = 1 - tanh(x)^2 (reuse the value).
  (void)x;
  return Dual();
}

// --- NN-specific activations ---

inline Dual sigmoid(const Dual& x) {
  // TODO(you): implement directly (don't compose from exp/+// for
  // numerical stability). sigmoid(x) = 1/(1+exp(-x)),
  // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
  (void)x;
  return Dual();
}

inline Dual relu(const Dual& x) {
  // TODO(you): implement. Pick and DOCUMENT your subgradient convention at
  // exactly x == 0 (0 is the conventional choice) — write it here as a
  // comment once decided, and add a test case for it in test_dual.cc.
  (void)x;
  return Dual();
}

}  // namespace forward
}  // namespace autodiff
