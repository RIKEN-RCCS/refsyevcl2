/***
 * testmat.hpp
 * Fast method for generating the real symmetric eigenvalue problem
 * 2024-06-07 UCHINO Yuki
 */
#pragma once

namespace testmat {
//
void gen_eig(double *A,                // [local output] generated matrix
             const int *descA,         // descriptor
             double *d1,               // [global output] eigenvalues
             double *d2,               // [global output] eigenvalues (d = d1 + d2)
             const double cnd = 1.e10, // anticipated condition number
             const int mode   = 3);      // distribution of eigenvalues

} // namespace testmat
