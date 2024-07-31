/***
 * eval.hpp
 * Evaluate the error
 * 2024-06-12 UCHINO Yuki
 */

#pragma once
namespace eval {

void disp(char *Mat, double *A, int mxllda, int mxloca, int nb, int *descA);
void comperr(const double *X,  // computed result
             const int *descX, //
             double *d,        // computed result
             const double *d1, // high order part of exact eigenvalues
             const double *d2, // low order part of exact eigenvalues
             double &err_d,    //
             double &err_X,    //
             double *work);    //

} // namespace eval
