/***
 * util.hpp
 * Utility functions
 * 2024-06-07 UCHINO Yuki
 */
#pragma once
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>

namespace util {

// return 2^ceil(log2(abs(a)))
template <typename T> inline T npt(T a);
template <> inline double npt(double a) {
    constexpr double uinv  = 9007199254740992.0;
    constexpr double muinv = -9007199254740992.0;
    double b               = std::fma(uinv, a, a);  // a + a*u^-1
    b                      = std::fma(muinv, a, b); // b - a*u^-1
    b                      = std::fabs(b);
    if (b == 0) b = std::fabs(a);
    return b;
}
template <> inline float npt(float a) {
    constexpr float uinv  = 16777216.0;
    constexpr float muinv = -16777216.0;
    float b               = std::fma(uinv, a, a);  // a + a*u^-1
    b                     = std::fma(muinv, a, b); // b - a*u^-1
    b                     = std::fabs(b);
    if (b == 0) b = std::fabs(a);
    return b;
}

// return global index using C index (Not FORTRUN index!!)
// global_i = l2g(local_i, nb, myrow, nprow);
// global_j = l2g(local_j, nb, mycol, npcol);
inline int l2g(int i, int nb, int iproc, int nproc) {
    return (((i / nb) * nproc + iproc) * nb + i % nb);
}

} // namespace util
