/***
 * ir.cpp
 * Iterative refinement for symmetric eigenproblem
 * 2024-06-07 UCHINO Yuki
 */
#include "ir.hpp"
#include "cpair.hpp"
#include "error_free.hpp"
#include "eval.hpp"
#include "ozmm.hpp"
#include "util.hpp"
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <omp.h>

extern "C" {
#include "eigen_exa_interfaces.h"
int Cblacs_gridinfo(const int, int *, int *, int *, int *);
int numroc_(const int *, const int *, const int *, const int *, const int *);
void descinit_(int *, const int *, const int *, const int *, const int *, const int *, const int *, const int *, int *, int *);
void pdgemm_(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const int *, const int *, const double *, const int *, const int *, const int *, const double *, double *, const int *, const int *, const int *);
void pdgeadd_(const char *, const int *, const int *, const double *, const double *, const int *, const int *, const int *, const double *, double *, const int *, const int *, const int *);

void dsyevd_(const char *, const char *, const int *, double *, const int *, double *, double *, const int *, const int *, const int *, const int *);
void pdtradd_(const char *, const char *, const int *, const int *, double *, double *, const int *, const int *, const int *, double *, double *, const int *, const int *, const int *);
void pdsyrk_(const char *, const char *, const int *, const int *, double *, double *, const int *, const int *, const int *, double *, double *, const int *, const int *, const int *);
double pdlange_(const char *, const int *, const int *, const double *, const int *, const int *, const int *, double *);
}

namespace {

// d(I(i):I(i)+N(i)-1) for i=0,1,...,nJ: clustered eigenvalues (in/output: FORTRUN index)
template <typename T> int findCL(const int n,
                                 T *d,
                                 const int idx,
                                 const T omega,
                                 int *I,
                                 int *N) {
    int nJ    = 0;
    int n2    = 0;
    int j     = idx - 1;
    T *tmpd   = d + j;
    bool *tmp = new bool[n + 1];

    tmp[0] = false;
    tmp[n] = false;
#pragma omp parallel for
    for (int i = 1; i < n; i++) {
        tmp[i] = (fabs(tmpd[i - 1] - tmpd[i]) <= omega);
    }

    for (int i = 1; i <= n; i++) {
        if (tmp[i] && (!tmp[i - 1])) {
            I[nJ] = j + i;
            nJ++;
        }
        if ((!tmp[i]) && tmp[i - 1]) {
            N[n2] = j + i;
            n2++;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < nJ; i++) N[i] -= (I[i] - 1);

    delete[] tmp;
    return nJ;
}

// compute median of vec[begin],...,vec[begin+n-1] (input: C index)
inline double median(double *vec, int n, int begin) {
    return (n & 1) ? vec[begin + n / 2] : (vec[begin + n / 2] + vec[begin + n / 2 - 1]) * 0.5;
}

} // namespace

void mpi_my_sum(void *in_, void *inout_, int *len, MPI_Datatype *dtype) {
    double *in    = (double *)in_;
    double *inout = (double *)inout_;

    for (int i = *len - 1; i >= 0; --i) {
        if (in[i] != inout[i]) inout[i] += in[i];
    }
}

// void mpi_my_sum_cpair(void *in_, void *inout_, int *len, MPI_Datatype *dtype) {
//     double *in    = (double *)in_;
//     double *inout = (double *)inout_;
//     int n         = *len / 2;

//     for (int i = n - 1; i >= 0; --i) {
//         cpair::add222(in[i], in[i + n], inout[i], inout[i + n], inout[i], inout[i + n]);
//     }
// }

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
            double *work      // [workspace] mxllda * mxloca * 3 + mxloca*2
#if defined(TIMING_REF)
            ,
            double *time
#endif
) {
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[0] = MPI_Wtime();
#endif
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(descA[1], &nprow, &npcol, &myrow, &mycol);                // get grid info
    int constexpr zero      = 0;                                              //
    int constexpr one       = 1;                                              //
    const int n             = descA[2];                                       // global size
    const int nb            = descA[5];                                       // block syclic size (columns)
    const int mxllda        = descA[8];                                       // leading dimension of local matrix
    const int mxloca        = numroc_(&descA[3], &nb, &mycol, &zero, &npcol); // #columns of local matrix
    char constexpr TT       = 'T';
    char constexpr NN       = 'N';
    double constexpr done   = 1.0;
    double constexpr dzero  = 0.0;
    double constexpr dmhalf = -0.5;
    MPI_Comm MPI_COMM_COLUMN;
    MPI_Comm_split(MPI_COMM_WORLD, mycol, 0, &MPI_COMM_COLUMN);
    MPI_Op MPI_MY_SUM;
    MPI_Op_create(mpi_my_sum, 1, &MPI_MY_SUM);

    //=====
    // Do NOT change the order of followings:
    //=====
    int offset = 0;
    double *X1 = R;
    double *X2 = F;
    double *X3 = work;
    offset += mxllda * mxloca;
    double *AX1 = work + offset;
    offset += mxllda * mxloca;
    double *AX2 = work + offset;
    offset += mxllda * mxloca;
    double *dS = work + offset;
    offset += mxloca;
    double *dR = work + offset;
    offset += mxloca;

    //=====
    // workX1 + workX2 := A*X
    //=====
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[1] = MPI_Wtime();
#endif
    error_free::split3_B(X, X1, X2, X3, descX, AX1);
    MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[1] = MPI_Wtime() - time[1];
#endif

#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[2] = MPI_Wtime();
#endif
    ozmm::ozgemm(A1, A2, A3, descA, X, X1, X2, X3, descX, AX1, AX2, descX);
    MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[2] = MPI_Wtime() - time[2];
#endif

    //=====
    // dS := diag(X*A*X)', i.e., dS(j) := sum_i( X(i,j)*AX(i,j) ),
    // dR := diag(X*X)'  , i.e., dR(j) := sum_i( X(i,j)*X(i,j) ),
    // d  := dS./dR
    // dR := 1-dR
    // AX_XD := A*X - X*D
    // omega := 4*sum(|AX_XD|,'all')/n
    //=====
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[3] = MPI_Wtime();
#endif
#pragma omp parallel for
    for (int lj = 0; lj < mxloca; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol);
        if (n <= gj) {
            dR[lj] = 0.0;
            dS[lj] = 0.0;
            continue;
        }
        double t1 = 0.0;
        double t2 = 0.0;
        double t3 = 0.0;
        double t4 = 0.0;
#pragma omp simd reduction(+ : t1) reduction(+ : t2) reduction(+ : t3) reduction(+ : t4)
        for (int li = 0; li < mxllda; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow);
            if (n <= gi) continue;
            double tmp1, tmp2;
            error_free::two_prod<double>(X[lj * mxllda + li], X[lj * mxllda + li], tmp1, tmp2);                   // tmp := X(i,j)*X(i,j)
            cpair::add222<double>(t1, t2, tmp1, tmp2, t1, t2);                                                    // dR(j) += tmp
            cpair::mul122<double>(X[lj * mxllda + li], AX1[lj * mxllda + li], AX2[lj * mxllda + li], tmp1, tmp2); // tmp := X(i,j)*AX(i,j)
            cpair::add222<double>(t3, t4, tmp1, tmp2, t3, t4);                                                    // dS(j) += tmp
        }
        dR[lj] = t1 + t2;
        dS[lj] = t3 + t4;
    }
    MPI_Allreduce(MPI_IN_PLACE, dS, mxloca * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_COLUMN);

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        d[i] = 0.0;
    }

    double *AX_XD = AX1;
    double omega  = 0.0;
#pragma omp parallel for reduction(+ : omega)
    for (int lj = 0; lj < mxloca; ++lj) {
        const auto gj = util::l2g(lj, nb, mycol, npcol);
        if (n <= gj) continue;
        d[gj]       = dS[lj] / dR[lj];      // x_(j)'*A*x_(j) / x_(j)'*x_(j)
        dR[lj]      = (1.0 - dR[lj]) * 0.5; // (1-dR)./2
        double tmpd = -d[gj];
#pragma omp simd reduction(+ : omega)
        for (int li = 0; li < mxllda; ++li) {
            const auto gi = util::l2g(li, nb, myrow, nprow);
            if (n <= gi) continue;
            AX_XD[lj * mxllda + li] = std::fma(X[lj * mxllda + li], tmpd, AX1[lj * mxllda + li] + AX2[lj * mxllda + li]); // AX-XD
            omega += std::fabs(AX_XD[lj * mxllda + li]);                                                                  // sum(|AX-XD|,'all')
        }
    }
    d[n] = omega;

    // Allreduce for d and omega (= d[n])
    MPI_Allreduce(MPI_IN_PLACE, d, n + 1, MPI_DOUBLE, MPI_MY_SUM, MPI_COMM_WORLD);

    omega = d[n] * 4.0 / n;
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[3] = MPI_Wtime() - time[3];
#endif

    //=====
    // F := X'*AX_XD
    //=====
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[4] = MPI_Wtime();
#endif
    pdgemm_(&TT, &NN, &n, &n, &n, &done, X, &one, &one, descX, AX_XD, &one, &one, descX, &dzero, F, &one, &one, descX);
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[4] = MPI_Wtime() - time[4];
#endif

    //=====
    // find clusters: d(I(i):I(i)+N(i)-1) with FORTRUN index
    //=====
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[5] = MPI_Wtime();
#endif
    int nJ = findCL<double>(n, d, one, omega, iCL, nCL);
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[5] = MPI_Wtime() - time[5];
#endif

#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[6] = MPI_Wtime();
#endif
    //=====
    // R(iCL,iCL) := X(:,iCL)'*X(:,iCL) for clusters
    //=====
    for (int i = 0; i < nJ; i++) {
        pdgemm_(&TT, &NN, &nCL[i], &nCL[i], &n, &done, X, &one, &iCL[i], descX, X, &one, &iCL[i], descX, &dzero, R, &iCL[i], &iCL[i], descX);
    }
    MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[6] = MPI_Wtime() - time[6];
#endif

    //=====
    // Compute E
    //=====
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[7] = MPI_Wtime();
#endif
    double *E = X3;
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
                const double gap = d[gj] - d[gi];
                if (gi == gj) {
                    E[lj * mxllda + li] = dR[lj];
                } else if (std::fabs(gap) > omega) {
                    E[lj * mxllda + li] = F[lj * mxllda + li] / gap;
                } else {
                    E[lj * mxllda + li] = dmhalf * R[lj * mxllda + li];
                }
            }
        }
    }
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[7] = MPI_Wtime() - time[7];
#endif

    //=====
    // Update X := X+X*E
    //=====
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[8] = MPI_Wtime();
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    pdgemm_(&NN, &NN, &n, &n, &n, &done, X, &one, &one, descX, E, &one, &one, descX, &dzero, AX2, &one, &one, descX);
    MPI_Barrier(MPI_COMM_WORLD);
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
                X[lj * mxllda + li] += AX2[lj * mxllda + li];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
    time[8] = MPI_Wtime() - time[8];
#endif

    MPI_Op_free(&MPI_MY_SUM);

#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[0] = MPI_Wtime() - time[0];
#endif
    return nJ;
}

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
              double *work,     // [workspace] mxllda * mxloca * 3 + mxloca*2 + n + 1 + 6 * 128 + 2 * 128 * 128
              int *iwork,       // [workspace] 128 * 5 + 3
              int *descExa,     // descriptor for EigenExa
              int *info         // infomation for eigenexa
#if defined(TIMING_REF)
              ,
              double *time
#endif
) {
#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[9]  = MPI_Wtime();
    time[10] = 0.0;
    time[11] = 0.0;
    time[12] = 0.0;
    time[13] = 0.0;
#endif
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(descA[1], &nprow, &npcol, &myrow, &mycol);        // get grid info
    int constexpr zero     = 0;                                       //
    int constexpr one      = 1;                                       //
    const int n            = descA[2];                                // global size
    const int nb           = descA[5];                                // block syclic size (columns)
    const int mxllda       = descA[8];                                // leading dimension of local matrix
    const int mxloca       = numroc_(&n, &nb, &mycol, &zero, &npcol); // #columns of local matrix
    char constexpr NN      = 'N';
    double constexpr done  = 1.0;
    double constexpr dzero = 0.0;

    // for small cluster
    int nb_small      = std::min(128, mxllda);
    char constexpr VV = 'V';
    char constexpr UU = 'U';
    int mxllda_small  = numroc_(&n, &nb_small, &myrow, &zero, &nprow);
    int mxloca_small  = numroc_(&n, &nb_small, &mycol, &zero, &npcol);
    int descSmall[9];
    descinit_(descSmall, &n, &n, &nb_small, &nb_small, &zero, &zero, &descA[1], &mxllda_small, info);

    // for eigen exa
    int nb_forward  = 48;
    int nb_backward = 128;
    char AA         = 'A';
    int mxllda_exa, mxloca_exa;
    eigen_get_matdims(&n, &mxllda_exa, &mxloca_exa);
    double *A_exa = work;
    double *X_exa = work + mxllda_exa * mxloca_exa;
    double *d_exa = work + 2 * mxllda_exa * mxloca_exa;

    //=====
    // Initial guess
    // [X,d,E,F,iCL,nCL,nJ] = RefSyEv2(A,X);
    //=====
#if defined(TIMING_REF)
    const int nJ = ir::refsyev(A1, A2, A3, descA, X, descX, R, F, d, iCL, nCL, work, time);
#else
    const int nJ = ir::refsyev(A1, A2, A3, descA, X, descX, R, F, d, iCL, nCL, work);
#endif

    if (nJ == 0) {
#if defined(TIMING_REF)
        MPI_Barrier(MPI_COMM_WORLD);
        time[9] = MPI_Wtime() - time[9];
#endif
        return nJ;
    }

    //=====
    // Improve for each cluster
    //=====
    for (int i = 0; i < nJ; ++i) {
        // eigs. d[begin],...,d[end] are clustered
        int begin = iCL[i] - 1;
        int end   = begin + nCL[i] - 1;

        // median of clustered eigs.
        double d_median = median(d, nCL[i], begin);

        //=====
        // Compute F_cl + 2*(I-E_cl)*(D-median(d)*I) (= X_cl' * (A-d_median*I) * X_cl)
        //=====
#if defined(TIMING_REF)
        MPI_Barrier(MPI_COMM_WORLD);
        double time_tmp = MPI_Wtime();
#endif
#pragma omp parallel
        {
#pragma omp for
            for (int lj = 0; lj < mxloca; ++lj) {
                const auto gj = util::l2g(lj, nb, mycol, npcol);
                if (gj < begin || end < gj) continue;
                const auto dj_sft = d[gj] - d_median; // (d[j] - median(d))
#pragma omp simd
                for (int li = 0; li < mxllda; ++li) {
                    const auto gi = util::l2g(li, nb, myrow, nprow);
                    if (gi < begin || end < gi) continue;
                    R[lj * mxllda + li] = fma(R[lj * mxllda + li], dj_sft, F[lj * mxllda + li]); // F(J,J) + (X(:,J)'*X(:,J)) * (D - muk*I)
                }
            }
        }
#if defined(TIMING_REF)
        MPI_Barrier(MPI_COMM_WORLD);
        time[10] += MPI_Wtime() - time_tmp;
#endif

        if (nCL[i] > nb_small) {

            //=====
            // copy for eigen exa
            //=====
#if defined(TIMING_REF)
            MPI_Barrier(MPI_COMM_WORLD);
            time_tmp = MPI_Wtime();
#endif
            pdgeadd_(&NN, &nCL[i], &nCL[i], &done, R, &iCL[i], &iCL[i], descX, &dzero, A_exa, &one, &one, descExa);
#if defined(TIMING_REF)
            MPI_Barrier(MPI_COMM_WORLD);
            time[11] += MPI_Wtime() - time_tmp;
#endif

            //=====
            // run EigenExa X_exa * diag(d_exa) * X_exa' := A_exa
            //=====
            MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
            time_tmp = MPI_Wtime();
#endif
            eigen_sx(&nCL[i], &nCL[i], A_exa, &mxllda_exa, d_exa, X_exa, &mxllda_exa, &nb_forward, &nb_backward, &AA);
#if defined(TIMING_REF)
            MPI_Barrier(MPI_COMM_WORLD);
            time[12] += MPI_Wtime() - time_tmp;
#endif

            //=====
            // improve X_cl := X_cl*X_exa
            //=====
            MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
            time_tmp = MPI_Wtime();
#endif
            pdgemm_(&NN, &NN, &n, &nCL[i], &nCL[i], &done, X, &one, &iCL[i], descX, X_exa, &one, &one, descExa, &dzero, A_exa, &one, &iCL[i], descX);
            MPI_Barrier(MPI_COMM_WORLD);

        } else {

            //=====
            // copy for dsysvd
            //=====
#if defined(TIMING_REF)
            MPI_Barrier(MPI_COMM_WORLD);
            time_tmp = MPI_Wtime();
#endif
            pdgeadd_(&NN, &nCL[i], &nCL[i], &done, R, &iCL[i], &iCL[i], descX, &dzero, A_exa, &one, &one, descSmall);
#if defined(TIMING_REF)
            MPI_Barrier(MPI_COMM_WORLD);
            time[11] += MPI_Wtime() - time_tmp;
#endif

            //=====
            // run eig X_exa * diag(d_exa) * X_exa' := A_exa
            //=====
            MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
            time_tmp = MPI_Wtime();
#endif
            if (myrow == 0 && mycol == 0) {
                int lwork  = 1 + 6 * nCL[i] + 2 * nCL[i] * nCL[i];
                int liwork = 3 + 5 * nCL[i];
                dsyevd_(&VV, &UU, &nCL[i], A_exa, &mxllda_small, X_exa, d_exa, &lwork, iwork, &liwork, info);
            }
#if defined(TIMING_REF)
            MPI_Barrier(MPI_COMM_WORLD);
            time[12] += MPI_Wtime() - time_tmp;
#endif

            //=====
            // improve X_cl := X_cl*X_exa
            //=====
            MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
            time_tmp = MPI_Wtime();
#endif
            pdgemm_(&NN, &NN, &n, &nCL[i], &nCL[i], &done, X, &one, &iCL[i], descX, X_exa, &one, &one, descSmall, &dzero, A_exa, &one, &iCL[i], descX);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        pdgeadd_(&NN, &n, &nCL[i], &done, A_exa, &one, &iCL[i], descX, &dzero, X, &one, &iCL[i], descX);
#if defined(TIMING_REF)
        MPI_Barrier(MPI_COMM_WORLD);
        time[13] += MPI_Wtime() - time_tmp;
#endif
    }

#if defined(TIMING_REF)
    MPI_Barrier(MPI_COMM_WORLD);
    time[9] = MPI_Wtime() - time[9];
#endif
    return nJ;
}

} // namespace ir
