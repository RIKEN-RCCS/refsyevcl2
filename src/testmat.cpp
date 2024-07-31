/***
 * testmat.cpp
 * Fast method for generating the real symmetric eigenvalue problem
 * 2024-06-07 UCHINO Yuki
 */
#include "testmat.hpp"
#include "cpair.hpp"
#include "error_free.hpp"
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
}

namespace {
// generate singular values
template <typename T>
void randsvd(T *d,           // [global output] eigenvalues
             const int n,    // size of d
             const T cnd,    // anticipated condition number
             const int mode) // distribution of eigenvalues
{
    if (mode == 1) {

        d[n - 1]     = 1.0 / n;
        const T scal = 1.0 / cnd / n;
#pragma omp parallel for
        for (int i = 0; i < n - 1; ++i) d[i] = scal;

    } else if (mode == 2) {

        d[0]         = 1.0 / cnd / n;
        const T scal = 1.0 / n;
#pragma omp parallel for
        for (int i = 1; i < n; ++i) d[i] = scal;

    } else if (mode == 3) {

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            d[i] = std::pow(cnd, -(n - i - 1.0) / (n - 1.0));
            d[i] /= n;
        }

    } else if (mode == 4) {

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            d[i] = 1.0 - (1.0 - 1.0 / cnd) * (n - i - 1.0) / (n - 1.0);
            d[i] /= n;
        }
    }
}

} // namespace

namespace testmat {

void gen_eig(double *A,        // [local output] generated matrix
             const int *descA, // descriptor
             double *d1,       // [global output] eigenvalues
             double *d2,       // [global output] eigenvalues (d = d1 + d2)
             const double cnd, // anticipated condition number
             const int mode)   // distribution of eigenvalues
{
    int zero = 0;
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(descA[1], &nprow, &npcol, &myrow, &mycol);
    const int n      = descA[2];                                // size of A
    const int nb     = descA[5];                                // block syclic size
    const int mxllda = descA[8];                                // leading dimension of local matrix
    const int mxloca = numroc_(&n, &nb, &mycol, &zero, &npcol); // #columns of local matrix

    // generate d
    randsvd(d1 + n / 2, n / 2, cnd, mode); // generate target eigenvalues

    // split value
    int r         = n - 4;                              //
    int e         = std::ceil(std::log2(r));            // e = nextpow2(r); ( 2^e = npt(n-4) )
    int t         = r & -r;                             // t = bitand(r,-r,'int64'); ( Unit in the last non-zero place )
    double s      = 1.0 / t;                            // 1/t
    double sigma2 = 1.5 * util::npt<double>(d1[n - 1]); // 1.5 * npt(max(abs(d1)));

    //=====
    // Determine d/n
    //=====
#pragma omp parallel for
    for (int i = n / 2; i < n; ++i) {
        int di        = std::ceil(std::log2(std::fabs(d1[i]))); // 2^di = np2(d1[i])
        double sigma1 = std::scalbln(0.75 * s, di + e);         // 0.75 * np2(d1[i]) * np2(n-4) / ulnp(n-4)
        double sigma  = std::max(sigma1, sigma2);               // determine sigma
        d1[i]         = std::fabs(d1[i] + sigma) - sigma;       // extract high order part
        d1[n - 1 - i] = -d1[i];
    }

    //=====
    // Compute A (error-free)
    //=====
#pragma omp parallel
    {
#pragma omp for
        for (int lj = 0; lj < mxloca; ++lj) {
            const auto gj = util::l2g(lj, nb, mycol, npcol);
            if (n <= gj) continue;
#pragma omp simd
            for (int li = 0; li < mxllda; ++li) {
                const auto gi = util::l2g(li, nb, myrow, nprow);
                if (n <= gi) continue;
                if (gi == gj) A[lj * mxllda + li] = r * d1[gi];
                else A[lj * mxllda + li] = -2.0 * (d1[gi] + d1[gj]);
            }
        }
    }

    //=====
    // d1+d2 := d/n * n (error-free)
    //=====
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        error_free::two_prod<double>(n, d1[i], d1[i], d2[i]);
    }
}

} // namespace testmat
