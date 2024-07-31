
#include "error_free.hpp"
#include "eval.hpp"
#include "ir.hpp"
#include "testmat.hpp"
#include "util.hpp"
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <omp.h>

extern "C" {
#include "eigen_exa_interfaces.h"
void descinit_(int *, const int *, const int *, const int *, const int *, const int *, const int *, const int *, int *, int *);
void Cblacs_get(int, int, int *);
int Cblacs_gridinfo(const int, int *, int *, int *, int *);
int numroc_(const int *, const int *, const int *, const int *, const int *);
void pdgeadd_(const char *, const int *, const int *, const double *, const double *, const int *, const int *, const int *, const double *, double *, const int *, const int *, const int *);
}

int main(int argc, char *argv[]) {
    // set n
    int log2_n = 10;
    if (argc > 1) {
        log2_n = atoi(argv[1]);
    }
    int n = (int)std::pow(2.0, log2_n);

    // set cnd (0 <= log10(cnd) <= 1e15)
    int log10_cnd = 8;
    if (argc > 2) {
        log10_cnd = std::min(std::max(0, atoi(argv[2])), 15);
    }
    double cnd = std::pow(10.0, log10_cnd);

    // set mode (1 <= mode <= 5)
    int mode = 3;
    if (argc > 3) {
        mode = std::min(std::max(1, atoi(argv[3])), 5);
    }

    // set #iterations (1 <= kmax <= 8)
    int kmax = 5;
    if (argc > 4) {
        kmax = std::min(std::max(1, atoi(argv[4])), 8);
    }

    // set MPI
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    int nprocs;
    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // EigenExa & BLACS init
    eigen_init();
    int myrank, myrow, mycol, nprow, npcol;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int ictxt = eigen_get_blacs_context();
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // for EigenExa
    int info;
    int zero   = 0;
    int nb_exa = 1;
    int mxllda_exa, mxloca_exa;
    eigen_get_matdims(&n, &mxllda_exa, &mxloca_exa);
    int descExa[9];
    descinit_(descExa, &n, &n, &nb_exa, &nb_exa, &zero, &zero, &ictxt, &mxllda_exa, &info);

    // workspace
    int nb        = 1;
    int mxllda    = numroc_(&n, &nb, &myrow, &zero, &nprow);
    int mxloca    = numroc_(&n, &nb, &mycol, &zero, &npcol);
    int sizeExa   = mxllda_exa * mxloca_exa;
    int sizeMat   = mxllda * mxloca;
    int lwork     = sizeMat * 2 + 4 * n + 1 + std::max(sizeExa * 2 + 33537, sizeMat * 8 + mxloca * 2);
    int *iwork    = new int[2 * n + 643]();
    double *dwork = new double[lwork]();
#if defined(TIMING_REF)
    double *time1 = new double[kmax * 14]();
#endif
    int descMat[9];
    descinit_(descMat, &n, &n, &nb, &nb, &zero, &zero, &ictxt, &mxllda, &info);

    // disp
    int nb_forward  = 48;
    int nb_backward = 128;
    int mem         = eigen_memory_internal(&n, &mxllda_exa, &mxllda_exa, &nb_forward, &nb_backward);
    eigen_show_version();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0) {
        printf("\n---\n");
        printf("Cond         : %e\n", cnd);
        printf("Mode         : %d\n", mode);
        printf("GlobalSize   : %d\n", n);
        printf("NumThreads   : %d\n", omp_get_max_threads());
        printf("NumProcess   : [%d,%d]\n", nprow, npcol);
        printf("BlockSize    : [%d,%d]\n", nb, nb);
        printf("LocalSize    : [%d,%d]\n", mxllda, mxloca);
        printf("BlockSize_Exa: [%d,%d]\n", nb_exa, nb_exa);
        printf("LocalSize_Exa: [%d,%d]\n", mxllda_exa, mxloca_exa);
        printf("Mem_internal : %d\n", mem);
        printf("---\n\n");
    }

    // for all (+ n)
    int offset = 0;
    double *a  = dwork + offset; // (mxllda * mxloca)
    offset += sizeMat;
    double *x = dwork + offset; // (mxllda * mxloca)
    offset += sizeMat;
    double *d = dwork + offset; // (n+1)
    offset += n + 1;
    double *d1 = dwork + offset; // (n)
    offset += n;
    double *d2 = dwork + offset; // (n)
    offset += n;

    // + for exa
    int offset1   = offset;
    double *a_exa = dwork + offset1; // (mxllda_exa * mxloca_exa)
    offset1 += sizeExa;
    double *x_exa = dwork + offset1; // (mxllda_exa * mxloca_exa)
    offset1 += sizeExa;

    // for ir
    double *a1 = dwork + offset; // (mxllda * mxloca)
    offset += sizeMat;
    double *a2 = dwork + offset; // (mxllda * mxloca)
    offset += sizeMat;
    double *a3 = dwork + offset; // (mxllda * mxloca)
    offset += sizeMat;
    double *R = dwork + offset; // (mxllda * mxloca)
    offset += sizeMat;
    double *F = dwork + offset; // (mxllda * mxloca)
    offset += sizeMat;
    double *dwork_ir = dwork + offset; // (mxllda * mxloca * 3 + mxloca*2)

    int *iCL      = iwork;     // (n)
    int *nCL      = iwork + n; // (n)
    int *iwork_ir = iwork + 2 * n;

    // Generate a

    for (int timemean = 0; timemean < 4; timemean++) {

        MPI_Barrier(MPI_COMM_WORLD);
        testmat::gen_eig(a, descMat, d1, d2, cnd, mode);
        MPI_Barrier(MPI_COMM_WORLD);
        if (myrank == 0) {
            printf("gen_eig: done\n");
        }

        // run EigenExa
        int one      = 1;
        char AA      = 'A';
        char NN      = 'N';
        double done  = 1.0;
        double dzero = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);

        pdgeadd_(&NN, &n, &n, &done, a, &one, &one, descMat, &dzero, a_exa, &one, &one, descExa);
        MPI_Barrier(MPI_COMM_WORLD);
        if (myrank == 0) {
            printf("cast a -> a_exa: done\n");
        }

#if defined(TIMING_REF)
        double time_exa = MPI_Wtime();
#endif
        eigen_sx(&n, &n, a_exa, &mxllda_exa, d, x_exa, &mxllda_exa, &nb_forward, &nb_backward, &AA);
        MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
        time_exa = MPI_Wtime() - time_exa;
#endif

        // check error
        MPI_Barrier(MPI_COMM_WORLD);
        eigen_get_errinfo(&info);
        if (myrank == 0) {
            printf("info = %d\n\n", info);
        }

        // copy eigenexa -> normal
        MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
        double time_convert = MPI_Wtime();
#endif
        pdgeadd_(&NN, &n, &n, &done, x_exa, &one, &one, descExa, &dzero, x, &one, &one, descMat);
        MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
        time_convert = MPI_Wtime() - time_convert;
#endif

        // compute error
        double err_d = 0.0;
        double err_x = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        eval::comperr(x, descMat, d, d1, d2, err_d, err_x, dwork_ir);

        // for MATLAB
        if (myrank == 0) {
            printf("res_%d_%d = [\n %e, %e, 0;\n", log2_n, log10_cnd, err_d, err_x);
        }

        // split a
        MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
        double time_splitA = MPI_Wtime();
#endif
        error_free::split3_A(a, a1, a2, a3, descMat, dwork_ir);
        MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
        time_splitA = MPI_Wtime() - time_splitA;
#endif

        double err_d2;
        for (int i = 0; i < kmax; i++) {

            MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIMING_REF)
            int nJ = ir::refsyevcl(a1, a2, a3, descMat, x, descMat, R, F, d, iCL, nCL, dwork_ir, iwork_ir, descExa, &info, time1 + i * 14);
#else
            int nJ = ir::refsyevcl(a1, a2, a3, descMat, x, descMat, R, F, d, iCL, nCL, dwork_ir, iwork_ir, descExa, &info);
#endif

            MPI_Barrier(MPI_COMM_WORLD);
            eval::comperr(x, descMat, d, d1, d2, err_d2, err_x, dwork_ir);

            // for MATLAB
            MPI_Barrier(MPI_COMM_WORLD);
            if (myrank == 0) {
                printf(" %e, %e, %d;\n", err_d2, err_x, nJ);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if (err_d / err_d2 < 2) break;

            err_d = err_d2;
        }

        //  MATLAB
        if (myrank == 0) {
            printf("];\n\n");
        }

#if defined(TIMING_REF)
        MPI_Barrier(MPI_COMM_WORLD);
        if (myrank == 0) {
            printf("time_exa_%d_%d = %e;\n", log2_n, log10_cnd, time_exa);
            printf("time_convert_%d_%d = %e;\n", log2_n, log10_cnd, time_convert);
            printf("time_splitA_%d_%d = %e;\n", log2_n, log10_cnd, time_splitA);
            printf("time_refsyev_%d_%d = [\n", log2_n, log10_cnd);
            for (int j = 0; j < 14; j++) {
                for (int i = 0; i < kmax; i++) {
                    printf(" %e ", time1[j + i * 14]);
                }
                if (j == 0) printf("%% refsyev total\n");
                if (j == 1) printf("%% split X\n");
                if (j == 2) printf("%% ozaki scheme\n");
                if (j == 3) printf("%% d, omega, AX-XD\n");
                if (j == 4) printf("%% X'*(AX-XD)\n");
                if (j == 5) printf("%% find clusters\n");
                if (j == 6) printf("%% E(i,j) for cluster\n");
                if (j == 7) printf("%% E(i,j) otherwise\n");
                if (j == 8) printf("%% X+X*E\n");
                if (j == 9) printf("%% refsyevcl total\n");
                if (j == 10) printf("%% X_cl'*(A-mu*I)*X_cl by O(nJ^2) ops\n");
                if (j == 11) printf("%% convert into A_exa\n");
                if (j == 12) printf("%% EigenExa for A_exa\n");
                if (j == 13) printf("%% X_cl := X_cl*X_exa\n");
            }
            printf("];\n\n");
        }

        for (int j = 0; j < 14; j++) {
            for (int i = 0; i < kmax; i++) {
                time1[j + i * 14] = 0.0;
            }
        }
#endif
    }

#if defined(TIMING_REF)
    delete[] time1; // Release the memory
#endif

    // Finalization
    MPI_Barrier(MPI_COMM_WORLD);
    delete[] dwork; // Release the memory
    delete[] iwork; // Release the memory

    MPI_Barrier(MPI_COMM_WORLD);
    eigen_free();
    MPI_Finalize(); // Exit the MPI
    return (0);
}
