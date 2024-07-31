/***
 * ir.hpp
 * Iterative refinement for symmetric eigenproblem
 * 2024-06-07 UCHINO Yuki
 */
#pragma once

namespace ir {

//=====
// [X,d,E,F,iCL,nCL,nJ] = RefSyEv2(A,X); d[n] = omega
//=====
int refsyev(const double *A1, // [local input] mxllda * mxloca
            const double *A2, // [local input] mxllda * mxloca
            const double *A3, // [local input] mxllda * mxloca
            const int *descA, // descriptor
            double *X,        // [local in/output] mxllda * mxloca
            const int *descX, // descriptor
            double *R,        // [local workspace/output] mxllda * mxloca
            double *F,        // [local workspace/output] mxllda * mxloca
            double *d,        // [global output] length of n+1, d[n] := omega
            int *iCL,         // [global output] length of n, first index of clusters
            int *nCL,         // [global output] length of n, #of each clusters
            double *work      // [workspace] mxllda * mxloca * 3 + mxloca*2 + n
#if defined(TIMING_REF)
            ,
            double *time
#endif
);

//=====
// RefSyEvCL2
//=====
int refsyevcl(const double *A1, // [local input] mxllda * mxloca
              const double *A2, // [local input] mxllda * mxloca
              const double *A3, // [local input] mxllda * mxloca
              const int *descA, // descriptor
              double *X,        // [local in/output] mxllda * mxloca
              const int *descX, // descriptor
              double *R,        // [local workspace/output] mxllda * mxloca
              double *F,        // [local workspace/output] mxllda * mxloca
              double *d,        // [global output] length of n+1, d[n] := omega
              int *iCL,         // [global output] length of n, first index of clusters
              int *nCL,         // [global output] length of n, #of each clusters
              double *work,     // [workspace] mxllda * mxloca * 3 + mxloca*2 + n
              int *iwork,       // [workspace] 128 * 5 + 3
              int *descExa,     // descriptor for EigenExa
              int *info         // infomation for eigenexa
#if defined(TIMING_REF)
              ,
              double *time
#endif
);

} // namespace ir
