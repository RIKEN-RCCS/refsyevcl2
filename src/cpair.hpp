/***
 * cpair.hpp
 * Four arithmetic operations using cpair
 * 2024-06-07 UCHINO Yuki
 */
#pragma once
#include "error_free.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace cpair {

//=====
// addition
//=====
template <typename T>
inline void add112(const T a1,
                   const T b1,
                   T &c1,
                   T &c2) {
    error_free::two_sum<T>(a1, b1, c1, c2);
}

template <typename T>
inline void add121(const T a1,
                   const T b1,
                   const T b2,
                   T &c1) {
    c1 = a1 + b1 + b2;
}

template <typename T>
inline void add122(const T a1,
                   const T b1,
                   const T b2,
                   T &c1,
                   T &c2) {
    error_free::two_sum<T>(a1, b1, c1, c2);
    c2 += b2;
}

template <typename T>
inline void add211(const T a1,
                   const T a2,
                   const T b1,
                   T &c1) {
    c1 = a1 + b1 + a2;
}

template <typename T>
inline void add212(const T a1,
                   const T a2,
                   const T b1,
                   T &c1,
                   T &c2) {
    error_free::two_sum<T>(a1, b1, c1, c2);
    c2 += a2;
}

template <typename T>
inline void add221(const T a1,
                   const T a2,
                   const T b1,
                   const T b2,
                   T &c1) {
    c1 = a1 + b1 + a2 + b2;
}

template <typename T>
inline void add222(const T a1,
                   const T a2,
                   const T b1,
                   const T b2,
                   T &c1,
                   T &c2) {
    error_free::two_sum<T>(a1, b1, c1, c2);
    c2 += a2 + b2;
}

//=====
// subtraction
//=====
template <typename T>
inline void sub112(const T a1,
                   const T b1,
                   T &c1,
                   T &c2) {
    error_free::two_sub<T>(a1, b1, c1, c2);
}

template <typename T>
inline void sub121(const T a1,
                   const T b1,
                   const T b2,
                   T &c1) {
    c1 = a1 - b1 - b2;
}

template <typename T>
inline void sub122(const T a1,
                   const T b1,
                   const T b2,
                   T &c1,
                   T &c2) {
    error_free::two_sub<T>(a1, b1, c1, c2);
    c2 -= b2;
}

template <typename T>
inline void sub211(const T a1,
                   const T a2,
                   const T b1,
                   T &c1) {
    c1 = (a1 - b1) + a2;
}

template <typename T>
inline void sub212(const T a1,
                   const T a2,
                   const T b1,
                   T &c1,
                   T &c2) {
    error_free::two_sub<T>(a1, b1, c1, c2);
    c2 += a2;
}

template <typename T>
inline void sub221(const T a1,
                   const T a2,
                   const T b1,
                   const T b2,
                   T &c1) {
    c1 = ((a1 - b1) + a2) - b2;
}

template <typename T>
inline void sub222(const T a1,
                   const T a2,
                   const T b1,
                   const T b2,
                   T &c1,
                   T &c2) {
    error_free::two_sub<T>(a1, b1, c1, c2);
    c2 += a2 - b2;
}

//=====
// multiplication
//=====
template <typename T>
inline void mul112(const T a1,
                   const T b1,
                   T &c1,
                   T &c2) {
    error_free::two_prod<T>(a1, b1, c1, c2);
}

template <typename T>
inline void mul121(const T a1,
                   const T b1,
                   const T b2,
                   T &c1) {
    c1 = a1 * (b1 + b2);
}

template <typename T>
inline void mul122(const T a1,
                   const T b1,
                   const T b2,
                   T &c1,
                   T &c2) {
    error_free::two_prod<T>(a1, b1, c1, c2);
    c2 = std::fma(a1, b2, c2);
}

template <typename T>
inline void mul211(const T a1,
                   const T a2,
                   const T b1,
                   T &c1) {
    c1 = (a1 + a2) * b1;
}

template <typename T>
inline void mul212(const T a1,
                   const T a2,
                   const T b1,
                   T &c1,
                   T &c2) {
    error_free::two_prod<T>(a1, b1, c1, c2);
    c2 = std::fma(a2, b1, c2);
}

template <typename T>
inline void mul221(const T a1,
                   const T a2,
                   const T b1,
                   const T b2,
                   T &c1) {
    c1 = std::fma(a2, b1, std::fma(a1, b1, a1 * b2));
}

template <typename T>
inline void mul222(const T a1,
                   const T a2,
                   const T b1,
                   const T b2,
                   T &c1,
                   T &c2) {
    error_free::two_prod<T>(a1, b1, c1, c2);
    c2 = std::fma(a2, b1, std::fma(a1, b2, c2));
}

//=====
// division
//=====
template <typename T>
inline void div112(const T a1,
                   const T b1,
                   T &c1,
                   T &c2) {
    c1 = a1 / b1;
    c2 = std::fma(-b1, c1, a1) / b1;
}

template <typename T>
inline void div121(const T a1,
                   const T b1,
                   const T b2,
                   T &c1) {
    c1 = a1 / (b1 + b2);
}

template <typename T>
inline void div122(const T a1,
                   const T b1,
                   const T b2,
                   T &c1,
                   T &c2) {
    c1     = a1 / b1;
    auto p = std::fma(-b1, c1, a1);
    c2     = std::fma(-c1, b2, p) / (b1 + b2);
}

template <typename T>
inline void div211(const T a1,
                   const T a2,
                   const T b1,
                   T &c1) {
    c1 = (a1 + a2) / b1;
}

template <typename T>
inline void div212(const T a1,
                   const T a2,
                   const T b1,
                   T &c1,
                   T &c2) {
    c1 = a1 / b1;
    c2 = (std::fma(-b1, c1, a1) + a2) / b1;
}

template <typename T>
inline void div221(const T a1,
                   const T a2,
                   const T b1,
                   const T b2,
                   T &c1) {
    c1 = (a1 + a2) / (b1 + b2);
}

template <typename T>
inline void div222(const T a1,
                   const T a2,
                   const T b1,
                   const T b2,
                   T &c1,
                   T &c2) {
    c1     = a1 / b1;
    auto p = std::fma(-b1, c1, a1) + a2;
    c2     = std::fma(-c1, b2, p) / (b1 + b2);
}
} // namespace cpair
