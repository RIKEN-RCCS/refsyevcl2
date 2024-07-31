/***
 * error_free.cpp
 * Error-free transformations
 * 2024-06-07 UCHINO Yuki
 */
#include "error_free.hpp"
#include "util.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

extern "C" {
int Cblacs_gridinfo(const int, int *, int *, int *, int *);
int numroc_(const int *, const int *, const int *, const int *, const int *);
}

namespace error_free {

// A1+A2+A3 := A3(:,ja:ja+n-1) (for A of AB)
// assume A=A'
void split3_A(const double *A,  // local input
              double *A1,       // local output
              double *A2,       // local output
              double *A3,       // local output (A1,A2,A3 in R^(m * n))
              const int *descA, // descriptor
              double *rowNrm)   // local workspace (size of mxllda * (2 + numThreads))
{
    //-----
    // local scalars
    //-----
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(descA[1], &nprow, &npcol, &myrow, &mycol);       // get grid info
    int constexpr zero    = 0;                                       //
    const int m           = descA[2];                                // global size
    const int n           = descA[3];                                // global size
    const int nb          = descA[5];                                // block syclic size (columns)
    const int mxllda      = descA[8];                                // leading dimension of local matrix
    const int mxloca      = numroc_(&n, &nb, &mycol, &zero, &npcol); // #columns of local matrix
    double tau            = 0.75 * std::pow(2.0, std::ceil((53 + std::log2(n)) * 0.5));
    const int num_threads = omp_get_max_threads();
    double *rowNrm2       = rowNrm + mxllda;
    double *rowNrm3       = rowNrm2 + mxllda;
    MPI_Comm MPI_COMM_ROW;
    MPI_Comm_split(MPI_COMM_WORLD, myrow, 0, &MPI_COMM_ROW);

    //-----
    // Compute rowNrm(j) := max(A3(:,j))
    //-----
#pragma omp parallel for
    for (int li = mxllda * (2 + num_threads) - 1; li >= 0; --li) {
        rowNrm[li] = 0.0;
    }

    int constexpr blk = 72;
#pragma omp parallel
    {
        int my_thread = omp_get_thread_num();

#pragma omp for collapse(2) schedule(dynamic)
        for (int llj = 0; llj < mxloca; llj += blk) {
            for (int lli = 0; lli < mxllda; lli += blk) {
                const auto li_end = std::min(lli + blk, mxllda);
                const auto lj_end = std::min(llj + blk, mxloca);
                for (int lj = llj; lj < lj_end; ++lj) {
                    const auto gj = util::l2g(lj, nb, mycol, npcol);
                    if (n <= gj) continue;
#pragma omp simd
                    for (int li = lli; li < li_end; ++li) {
                        const auto gi = util::l2g(li, nb, myrow, nprow);
                        if (m <= gi) continue;
                        A3[lj * mxllda + li] = A[lj * mxllda + li];
                        const int idx        = li + my_thread * mxllda;
                        rowNrm3[idx]         = std::max(rowNrm3[idx], std::fabs(A3[lj * mxllda + li]));
                    }
                }
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int li = 0; li < mxllda; ++li) {
        const auto gi = util::l2g(li, nb, myrow, nprow);
        if (m <= gi) continue;
        for (int it = 0; it < num_threads; ++it) {
            const int idx = li + it * mxllda;
            rowNrm[li]    = std::max(rowNrm[li], rowNrm3[idx]);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, rowNrm, mxllda, MPI_DOUBLE, MPI_MAX, MPI_COMM_ROW);

    //-----
    // Split A3 =: A1+A3 & Compute rowNrm(j) >= norm(A3(:,j))
    //-----
#pragma omp parallel for
    for (int li = mxllda - 1; li >= 0; --li) {
        rowNrm[li] = tau * util::npt<double>(rowNrm[li]);
    }

#pragma omp parallel for
    for (int li = mxllda * num_threads - 1; li >= 0; --li) {
        rowNrm3[li] = 0.0;
    }

#pragma omp parallel
    {
        int my_thread = omp_get_thread_num();

#pragma omp for collapse(2) schedule(dynamic)
        for (int llj = 0; llj < mxloca; llj += blk) {
            for (int lli = 0; lli < mxllda; lli += blk) {
                const auto lj_end = std::min(llj + blk, mxloca);
                const auto li_end = std::min(lli + blk, mxllda);
                for (int lj = llj; lj < lj_end; ++lj) {
                    const auto gj = util::l2g(lj, nb, mycol, npcol);
                    if (n <= gj) continue;
#pragma omp simd
                    for (int li = lli; li < li_end; ++li) {
                        const auto gi = util::l2g(li, nb, myrow, nprow);
                        if (m <= gi) continue;
                        error_free::extract_scal<double>(rowNrm[li], A1[lj * mxllda + li], A3[lj * mxllda + li]); // A1 + A3 := A3
                        const int idx = li + my_thread * mxllda;
                        rowNrm3[idx]  = std::max(rowNrm3[idx], std::fabs(A3[lj * mxllda + li]));
                    }
                }
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int li = 0; li < mxllda; ++li) {
        const auto gi = util::l2g(li, nb, myrow, nprow);
        if (m <= gi) continue;
        for (int it = 0; it < num_threads; ++it) {
            const int idx = li + it * mxllda;
            rowNrm2[li]   = std::max(rowNrm2[li], rowNrm3[idx]);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, rowNrm2, mxllda, MPI_DOUBLE, MPI_MAX, MPI_COMM_ROW);

    //-----
    // Split A3 =: A2+A3
    //-----
#pragma omp parallel for
    for (int li = mxllda - 1; li >= 0; --li) {
        rowNrm2[li] = tau * util::npt<double>(rowNrm2[li]);
    }
#pragma omp parallel for
    for (int lj = 0; lj < mxloca; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol); // global j
        if (n <= gj) continue;
#pragma omp simd
        for (int li = 0; li < mxllda; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow); // global i
            if (m <= gi) continue;
            error_free::extract_scal<double>(rowNrm2[li], A2[lj * mxllda + li], A3[lj * mxllda + li]); // A2 + A3 := A3
        }
    }
}

// A1+A2+A3 := A3(:,ja:ja+n-1) (for B of AB)
void split3_B(const double *A,  // local input
              double *A1,       // local output
              double *A2,       // local output
              double *A3,       // local output (A1,A2,A3 in R^(m * n))
              const int *descA, // descriptor
              double *colNrm)   // local workspace (size of mxloca * 2)
{
    //-----
    // local scalars
    //-----
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(descA[1], &nprow, &npcol, &myrow, &mycol);    // get grid info
    int constexpr zero = 0;                                       //
    const int m        = descA[2];                                // global size
    const int n        = descA[3];                                // global size
    const int nb       = descA[5];                                // block syclic size (columns)
    const int mxllda   = descA[8];                                // leading dimension of local matrix
    const int mxloca   = numroc_(&n, &nb, &mycol, &zero, &npcol); // #columns of local matrix
    double tau         = 0.75 * std::pow(2.0, std::ceil((53 + std::log2(m)) * 0.5));
    double *colNrm2    = colNrm + mxloca;
    MPI_Comm MPI_COMM_COLUMN;
    MPI_Comm_split(MPI_COMM_WORLD, mycol, 0, &MPI_COMM_COLUMN);

    //-----
    // Compute colNrm(j) := max(A3(:,j))
    //-----
#pragma omp parallel for
    for (int lj = 0; lj < mxloca; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol);
        if (n <= gj) {
            colNrm[lj] = 0.0;
            continue;
        }
        double tmp = 0.0;
#pragma omp simd reduction(max : tmp)
        for (int li = 0; li < mxllda; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow);
            if (m <= gi) continue;
            A3[lj * mxllda + li] = A[lj * mxllda + li];
            tmp                  = std::max(std::fabs(A3[lj * mxllda + li]), tmp);
        }
        colNrm[lj] = tau * util::npt<double>(tmp);
    }
    MPI_Allreduce(MPI_IN_PLACE, colNrm, mxloca, MPI_DOUBLE, MPI_MAX, MPI_COMM_COLUMN);

    //-----
    // Split A3 =: A1+A3 & Compute colNrm(j) >= norm(A3(:,j))
    //-----
#pragma omp parallel for
    for (int lj = 0; lj < mxloca; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol);
        if (n <= gj) {
            colNrm2[lj] = 0.0;
            continue;
        }
        double tmp = 0.0;
#pragma omp simd reduction(max : tmp)
        for (int li = 0; li < mxllda; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow);
            if (m <= gi) continue;
            error_free::extract_scal<double>(colNrm[lj], A1[lj * mxllda + li], A3[lj * mxllda + li]); // A1 + A3 := A3
            tmp = std::max(std::fabs(A3[lj * mxllda + li]), tmp);
        }
        colNrm2[lj] = tau * util::npt<double>(tmp);
    }
    MPI_Allreduce(MPI_IN_PLACE, colNrm2, mxloca, MPI_DOUBLE, MPI_MAX, MPI_COMM_COLUMN);

    //-----
    // Split A3 =: A2+A3
    //-----
#pragma omp parallel for // schedule(dynamic)
    for (int lj = 0; lj < mxloca; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol);
        if (n <= gj) continue;
#pragma omp simd
        for (int li = 0; li < mxllda; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow);
            if (m <= gi) continue;
            error_free::extract_scal<double>(colNrm2[lj], A2[lj * mxllda + li], A3[lj * mxllda + li]); // A2 + A3 := A3
        }
    }
}

} // namespace error_free
