/***
 * ozmm.hpp
 * Highly accurate matrix multiplication using cpair and Ozaki's scheme
 * 2024-06-07 UCHINO Yuki
 */
#pragma once

namespace ozmm {

// void ozgemm(double *A1,  // input matrix
//             double *A2,  // input matrix
//             double *A3,  // input matrix
//             int *DESCA,  // descripter for A
//             double *B,   // input matrix
//             double *B1,  // input matrix, workspace
//             double *B2,  // input matrix, workspace
//             double *B3,  // input matrix, workspace
//             int *DESCB,  // descripter for B
//             double *C1,  // output
//             double *C2,  // output
//             int *DESCC); // descripter for C
void ozgemm(const double *A1,  // input matrix
            const double *A2,  // input matrix
            const double *A3,  // input matrix
            const int *DESCA,  // descripter for A
            const double *B,   // input matrix
            double *B1,        // input matrix, workspace
            double *B2,        // input matrix, workspace
            double *B3,        // input matrix, workspace
            const int *DESCB,  // descripter for B
            double *C1,        // output
            double *C2,        // output
            const int *DESCC); // descripter for C

} // namespace ozmm
