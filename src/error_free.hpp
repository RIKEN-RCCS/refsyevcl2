/***
 * error_free.hpp
 * Error-free transformations
 * 2024-06-07 UCHINO Yuki
 */
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace error_free {

// c+d := a+b
template <typename T>
inline void two_sum(T a, T b, T &c, T &d) {
    c   = a + b;
    T s = c - a;
    // T t = b - s;
    // T u = c - s;
    // d   = (a - u) + t;
    b -= s;
    s -= c;
    s += a;
    d = s + b;
}

// c+d := a-b
template <typename T>
inline void two_sub(T a, T b, T &c, T &d) {
    c   = a - b;
    T s = c - a;
    // T t = b + s;
    // T u = c - s;
    // d   = (a - u) - t;
    b += s;
    s -= c;
    s += a;
    d = s - b;
}

// c+d := a+b
template <typename T>
inline void fast_two_sum(T a, T b, T &c, T &d) {
    c   = a + b;
    T t = a - c;
    d   = t + b;
}

// c+d := a*b
template <typename T>
inline void two_prod(T a, T b, T &c, T &d) {
    c = a * b;
    d = std::fma(a, b, -c);
}

// a1 + a2 := a2
template <typename T>
inline void extract_scal(T sigma, T &a1, T &a2) {
    a1 = std::fabs(a2 + sigma) - sigma;
    a2 -= a1;
}

void split3_A(const double *A,  // local input
              double *A1,       // local output
              double *A2,       // local output
              double *A3,       // local output (A1,A2,A3 in R^(m * n))
              const int *descA, // descriptor
              double *rowNrm);  // local workspace (size of mxllda * (2 + numThreads))

// A1+A2+A3 := A3(:,ja:ja+n-1) (for B of AB)
void split3_B(const double *A,  // local input
              double *A1,       // local output
              double *A2,       // local output
              double *A3,       // local output (A1,A2,A3 in R^(m * n))
              const int *descA, // descriptor
              double *colNrm);  // local workspace (size of mxloca * 1)

} // namespace error_free
