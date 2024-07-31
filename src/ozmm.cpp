/***
 * ozmm.cpp
 * Highly accurate matrix multiplication using cpair and Ozaki's scheme
 * 2024-06-07 UCHINO Yuki
 */
#include "ozmm.hpp"
#include "error_free.hpp"
// #include "scsumma25d.h"
#include "util.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

extern "C" {
int Cblacs_gridinfo(const int, int *, int *, int *, int *);
int numroc_(const int *, const int *, const int *, const int *, const int *);
void pdgemm_(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const int *, const int *, const double *, const int *, const int *, const int *, const double *, double *, const int *, const int *, const int *);
}

namespace ozmm {

// C1 + C2 := A*B,
void ozgemm(const double *A1, // input matrix
            const double *A2, // input matrix
            const double *A3, // input matrix
            const int *DESCA, // descripter for A
            const double *B,  // input matrix
            double *B1,       // input matrix, workspace
            double *B2,       // input matrix, workspace
            double *B3,       // input matrix, workspace
            const int *DESCB, // descripter for B
            double *C1,       // output
            double *C2,       // output
            const int *DESCC) // descripter for C
{
    char NN = 'N';
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(DESCA[1], &nprow, &npcol, &myrow, &mycol);
    double done  = 1.0;
    double dzero = 0.0;
    int nb       = DESCA[5];
    int mxllda   = DESCA[8];
    int one      = 1;
    int zero     = 0;
    int n        = DESCA[2]; // global size
    int mxloca   = numroc_(&DESCA[3], &nb, &mycol, &zero, &npcol);

    // C1 := A1*B1
    // C2 := A1*B2 + A2*B1
    pdgemm_(&NN, &NN, &n, &n, &n, &done, A1, &one, &one, DESCA, B1, &one, &one, DESCB, &dzero, C1, &one, &one, DESCC);
    pdgemm_(&NN, &NN, &n, &n, &n, &done, A1, &one, &one, DESCA, B2, &one, &one, DESCB, &dzero, C2, &one, &one, DESCC);
    pdgemm_(&NN, &NN, &n, &n, &n, &done, A2, &one, &one, DESCA, B1, &one, &one, DESCB, &done, C2, &one, &one, DESCC);
    MPI_Barrier(MPI_COMM_WORLD);

// [c1,c2] = FastTwoSum(c1,c2)
// B2 := B2 + B3
#pragma omp parallel for
    for (int lj = 0; lj < mxloca; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol);
        if (n <= gj) continue;
#pragma omp simd
        for (int li = 0; li < mxllda; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow);
            if (n <= gi) continue;
            error_free::fast_two_sum<double>(C1[lj * mxllda + li], C2[lj * mxllda + li], C1[lj * mxllda + li], C2[lj * mxllda + li]);
            B2[lj * mxllda + li] += B3[lj * mxllda + li];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // c2 = c2 + a1*b3 + a2*(b2+b3) + a3*(b1+b2+b3)
    pdgemm_(&NN, &NN, &n, &n, &n, &done, A1, &one, &one, DESCA, B3, &one, &one, DESCB, &done, C2, &one, &one, DESCC); // c2 += a1*b3
    pdgemm_(&NN, &NN, &n, &n, &n, &done, A2, &one, &one, DESCA, B2, &one, &one, DESCB, &done, C2, &one, &one, DESCC); // c2 += a2*(b2+b3)
    pdgemm_(&NN, &NN, &n, &n, &n, &done, A3, &one, &one, DESCA, B, &one, &one, DESCB, &done, C2, &one, &one, DESCC);  // c2 += a3*(b1+b2+b3)
}

} // namespace ozmm
