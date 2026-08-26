// Style reference only — not part of the build (not added to
// CMakeLists.txt, not compiled, not included by any header).
//
// This is the template-based structure CLAUDE.md points to when it says
// "structure new autodiff/nn code as templates": a class templated on its
// scalar type, Doxygen comments on every public member (including a
// one-line @brief plus @tparam/@param/@return where relevant), an internal
// `detail` namespace for helpers that are implementation details rather
// than public API, compound-assignment operators (+=, -=, *=, /=)
// implemented first with the binary operators built on top of them (or, as
// here, free binary operators implemented directly when that reads
// clearer), and elementary math functions as free functions that pull in
// the right overload via `using std::foo;` (ADL) so the same function
// template works for `double`, `float`, or another autodiff type nested
// inside it.
//
// include/autodiff/Dual.hh has since been retrofitted to this same
// templated style (autodiff::forward::Dual<T>) — see
// docs/notes/implementation-roadmap.md Stage 1/3 for the history. Consult
// this file for *how to write* new templated classes (Layer,
// NeuralNetwork, a future reverse-mode Var, ...); its naming
// (AD::Dual<T>::dual(), operator=(T), set(), ...) is illustrative only and
// doesn't match Dual.hh's actual API (value()/derivative(),
// variable()/constant(), implicit T constructor) — Dual.hh's API is fixed
// by docs/notes/implementation-roadmap.md Stage 1, not by this file.

#pragma once

#ifndef DUAL_HH
#define DUAL_HH

#include <cmath>
#include <ostream>

/**
 * @file dual.hh
 * @brief Dual-number implementation for forward-mode automatic differentiation.
 */

namespace AD {

  namespace detail {

    /**
     * @brief Returns the square of a value.
     *
     * @tparam T value type.
     * @param x value to square.
     * @return T square of @p x.
     */
    template <typename T>
    constexpr inline T square( T const & x ) noexcept(noexcept(x * x)) {
      return x * x;
    }

    /**
     * @brief Returns the constant factor in the derivative of `erf`.
     *
     * @tparam T value type.
     * @return T value of `2 / sqrt(pi)`.
     */
    template <typename T>
    inline T erf_scale() {
      using std::acos;
      using std::sqrt;
      return T(2) / sqrt(acos(T(-1)));
    }

  }

  /**
   * @brief Dual number of the form `a + b eps`.
   *
   * A dual number stores both the function value (`a`) and the first
   * derivative with respect to an independent variable (`b`).
   *
   * @tparam T scalar coefficient type.
   */
  template <typename T>
  class Dual {
  public:

    //! Underlying scalar type.
    using value_type = T;

  private:

    T _value = T(0);
    T _dual  = T(0);

  public:

    /**
     * @brief Builds the zero dual number.
     */
    constexpr Dual() noexcept = default;

    /**
     * @brief Builds a dual number by copy.
     */
    constexpr Dual( Dual const & ) noexcept = default;

    /**
     * @brief Builds a dual number by move.
     */
    constexpr Dual( Dual && ) noexcept = default;

    /**
     * @brief Builds a dual number from primal and dual parts.
     *
     * @param value primal value.
     * @param dual dual coefficient.
     */
    constexpr explicit Dual( T value, T dual = T(0) ) noexcept
    : _value(value)
    , _dual(dual)
    {}

    /**
     * @brief Assigns a dual number by copy.
     *
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator = ( Dual const & ) noexcept = default;

    /**
     * @brief Assigns a dual number by move.
     *
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator = ( Dual && ) noexcept = default;

    /**
     * @brief Builds an independent variable.
     *
     * @param value variable value.
     * @return Dual dual number with unit initial derivative.
     */
    [[nodiscard]] static constexpr Dual variable( T value ) noexcept {
      return Dual(value, T(1));
    }

    /**
     * @brief Returns the primal value.
     *
     * @return T primal value.
     */
    [[nodiscard]] constexpr T value() const noexcept { return _value; }

    /**
     * @brief Returns the dual component.
     *
     * @return T derivative coefficient.
     */
    [[nodiscard]] constexpr T dual() const noexcept { return _dual; }

    /**
     * @brief Sets both components of the dual number.
     *
     * @param value new primal value.
     * @param dual new dual coefficient.
     */
    constexpr void set( T value, T dual ) noexcept {
      _value = value;
      _dual  = dual;
    }

    /**
     * @brief Assigns a scalar value.
     *
     * The dual part is reset to zero.
     *
     * @param value new primal value.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator = ( T value ) noexcept {
      _value = value;
      _dual  = T(0);
      return *this;
    }

    /**
     * @brief Adds a dual number.
     *
     * @param rhs addend.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator += ( Dual const & rhs ) noexcept {
      _value += rhs._value;
      _dual  += rhs._dual;
      return *this;
    }

    /**
     * @brief Adds a scalar.
     *
     * @param rhs scalar addend.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator += ( T rhs ) noexcept {
      _value += rhs;
      return *this;
    }

    /**
     * @brief Subtracts a dual number.
     *
     * @param rhs subtrahend.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator -= ( Dual const & rhs ) noexcept {
      _value -= rhs._value;
      _dual  -= rhs._dual;
      return *this;
    }

    /**
     * @brief Subtracts a scalar.
     *
     * @param rhs scalar subtrahend.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator -= ( T rhs ) noexcept {
      _value -= rhs;
      return *this;
    }

    /**
     * @brief Multiplies by a dual number.
     *
     * @param rhs factor.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator *= ( Dual const & rhs ) noexcept {
      T const value = _value;
      T const dual  = _dual;
      _value = value * rhs._value;
      _dual  = dual * rhs._value + value * rhs._dual;
      return *this;
    }

    /**
     * @brief Multiplies by a scalar.
     *
     * @param rhs scalar factor.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator *= ( T rhs ) noexcept {
      _value *= rhs;
      _dual  *= rhs;
      return *this;
    }

    /**
     * @brief Divides by a dual number.
     *
     * @param rhs divisor.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator /= ( Dual const & rhs ) noexcept {
      T const value = _value;
      T const dual  = _dual;
      T const denom = rhs._value * rhs._value;
      _value = value / rhs._value;
      _dual  = (dual * rhs._value - value * rhs._dual) / denom;
      return *this;
    }

    /**
     * @brief Divides by a scalar.
     *
     * @param rhs scalar divisor.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator /= ( T rhs ) noexcept {
      _value /= rhs;
      _dual  /= rhs;
      return *this;
    }

    /**
     * @brief Applies unary plus.
     *
     * @return Dual unchanged copy of the dual number.
     */
    [[nodiscard]] constexpr Dual operator + () const noexcept {
      return *this;
    }

    /**
     * @brief Applies unary minus.
     *
     * @return Dual negated dual number.
     */
    [[nodiscard]] constexpr Dual operator - () const noexcept {
      return Dual(-_value, -_dual);
    }

  };

  // ... binary operators, comparisons, ostream<<, and the elementary math
  // functions (sin/cos/exp/log/sqrt/pow/...) all follow the same pattern:
  // free `template <typename T>` functions taking `Dual<T> const &`,
  // pulling in the right std:: overload via `using std::foo;` so the same
  // template works whether T is float, double, or a nested Dual.

}

#endif
