/***
 * eval.cpp
 * Evaluate the error
 * 2024-06-12 UCHINO Yuki
 */

#include "eval.hpp"
#include "cpair.hpp"
#include "error_free.hpp"
#include "ozmm.hpp"
#include "util.hpp"
#include <cblas.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <omp.h>

extern "C" {
int Cblacs_gridinfo(const int, int *, int *, int *, int *);
int numroc_(const int *, const int *, const int *, const int *, const int *);
void descinit_(int *, const int *, const int *, const int *, const int *, const int *, const int *, const int *, int *, int *);
double pdlange_(const char *, const int *, const int *, const double *, const int *, const int *, const int *, double *);
}

namespace {
double derr(const int n, double *d, const double *exact_d1, const double *exact_d2) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        double tmp1, tmp2;
        cpair::sub122<double>(d[i], exact_d1[i], exact_d2[i], tmp1, tmp2);
        cpair::div221<double>(tmp1, tmp2, exact_d1[i], exact_d2[i], d[i]);
    }
    int idx = cblas_idamax(n, d, 1); // d[idx] = max(abs(d))
    return std::fabs(d[idx]);
}
} // namespace

namespace eval {

void disp(char *Mat, double *A, int mxllda, int mxloca, int nb, int *descA) {
    int nprow, npcol, myrow, mycol;
    int n = descA[2];
    Cblacs_gridinfo(descA[1], &nprow, &npcol, &myrow, &mycol); // get grid info
    for (int lj = 0; lj < 5; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol);
        if (n <= gj) continue;
        for (int li = 0; li < 5; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow);
            if (n <= gi) continue;
            printf("%s(%d,%d) = %.16e;\n", Mat, gi + 1, gj + 1, A[lj * mxllda + li]);
        }
    }
};

void comperr(const double *X,  // computed result
             const int *descX, //
             double *d,        // computed result
             const double *d1, // high order part of exact eigenvalues
             const double *d2, // low order part of exact eigenvalues
             double &err_d,    //
             double &err_X,    //
             double *work)     //
{
    const int zero = 0;
    const int one  = 1;
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(descX[1], &nprow, &npcol, &myrow, &mycol);         // get grid info
    const int n      = descX[2];                                       // global size
    const int nb     = descX[5];                                       // block syclic size (columns)
    const int mxllda = descX[8];                                       // leading dimension of local matrix
    const int mxloca = numroc_(&descX[3], &nb, &mycol, &zero, &npcol); // #columns of local matrix
    const char MM    = 'M';

    // error in d
    err_d = derr(n, d, d1, d2);

    // exact X := I-2/n
    double Xdiag_1, Xdiag_2, Xoffdiag_1, Xoffdiag_2;
    cpair::div112<double>(2.0, n, Xoffdiag_1, Xoffdiag_2);                            // Xij := 2/n
    error_free::fast_two_sum<double>(Xoffdiag_1, Xoffdiag_2, Xoffdiag_1, Xoffdiag_2); //
    cpair::sub122<double>(1.0, Xoffdiag_1, Xoffdiag_2, Xdiag_1, Xdiag_2);             // Xii := 1-2/n
    error_free::fast_two_sum<double>(Xdiag_1, Xdiag_2, Xdiag_1, Xdiag_2);             //

    // error in X
    MPI_Barrier(MPI_COMM_WORLD);
#pragma omp parallel for schedule(dynamic)
    for (int lj = 0; lj < mxloca; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol);
        if (n <= gj) continue;
        for (int li = 0; li < mxllda; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow);
            if (n <= gi) continue;
            if (gi == gj) {
                work[lj * mxllda + li] = std::fabs((std::fabs(X[lj * mxllda + li]) - Xdiag_1) / Xdiag_1);
            } else {
                work[lj * mxllda + li] = std::fabs((std::fabs(X[lj * mxllda + li]) - Xoffdiag_1) / Xoffdiag_1);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    err_X = pdlange_(&MM, &n, &n, work, &one, &one, descX, NULL);

    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace eval
