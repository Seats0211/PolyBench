/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
        DATA_TYPE *alpha,
        DATA_TYPE *beta,
        DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
         DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
    if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
         DATA_TYPE alpha,
         DATA_TYPE beta,
         DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
         DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
         DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++)
    C[i][j] *= beta;
    for (k = 0; k < _PB_NK; k++) {
#if defined(SE)
  const DATA_TYPE A_TH = SCALAR_VAL(1e-3);
  const DATA_TYPE NJ_TH = SCALAR_VAL(50.0); /* tune per problem size */
  int condA = (fabs(A[i][k]) > A_TH);      /* cheap predicate */
  int condB = ((k % 2) == 0);
  int condC = (((int)(B[k][0]*1000.0)) & 1);

  if (condA) {
    if (condB || (condC && (nj > NJ_TH))) {
      /* hot: full accumulation */
      #if defined(LU) && (UNROLL > 1)
      /* optional manual unrolling here */
      int jj = 0;
      for (; jj + (UNROLL - 1) < _PB_NJ; jj += UNROLL) {
        /* unrolled body (example for UNROLL==4) */
        C[i][jj+0] += alpha * A[i][k] * B[k][jj+0];
        C[i][jj+1] += alpha * A[i][k] * B[k][jj+1];
        C[i][jj+2] += alpha * A[i][k] * B[k][jj+2];
        C[i][jj+3] += alpha * A[i][k] * B[k][jj+3];
      }
      for (; jj < _PB_NJ; ++jj) C[i][jj] += alpha * A[i][k] * B[k][jj];
      #else
      for (j = 0; j < _PB_NJ; j++)
      C[i][j] += alpha * A[i][k] * B[k][j];
      #endif
    } else {
      /* condA true but subguard false -> reduced accumulation (step-2) */
      for (j = 0; j < _PB_NJ; j += 2) {
        DATA_TYPE tmp = alpha * A[i][k] * B[k][j];
        C[i][j] += tmp * 2.0;
        if (j + 1 < _PB_NJ)
          C[i][j + 1] += tmp * 2.0;  /* approximate using same tmp */
      }
    }
  } else {
    /* condA false: another branch with similar pattern */
    if (condC && (nj > NJ_TH)) {
      /* alternate hot: full */
      for (j = 0; j < _PB_NJ; j++)
      C[i][j] += alpha * A[i][k] * B[k][j];
    } else {
      /* cold: lighter approximate - skip this k */
      /* nothing */
    }
  }
#else
       for (j = 0; j < _PB_NJ; j++)
      C[i][j] += alpha * A[i][k] * B[k][j];
#endif
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
          POLYBENCH_ARRAY(C),
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
           alpha, beta,
           POLYBENCH_ARRAY(C),
           POLYBENCH_ARRAY(A),
           POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
