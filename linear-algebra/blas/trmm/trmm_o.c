/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trmm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trmm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (DATA_TYPE)((i+j) % m)/m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (DATA_TYPE)((n+(i-j)) % n)/n;
    }
 }

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trmm(int m, int n,
         DATA_TYPE alpha,
         DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
         DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;

#pragma scop
  for (i = 0; i < _PB_M; i++) {
    for (j = 0; j < _PB_N; j++) {
#if defined(SE)
      /* Tunable predicates / thresholds */
      const DATA_TYPE ALPHA_TH = SCALAR_VAL(1e-3);
      const DATA_TYPE N_TH     = SCALAR_VAL(64.0);

      int condA = (fabs(alpha) > ALPHA_TH);           /* cheap predicate */
      int condB = ((i + j) % 2 == 0);                 /* structural predicate */
      int condC = (((int)(A[i][0] * 1000.0)) & 1);    /* pseudo-random bit */

      if (condA) {
        if (condB || (condC && (m > N_TH))) {
          /* hot path */
          DATA_TYPE acc = B[i][j];
#if defined(LU) && (UNROLL > 1)
          int kk = i + 1;
          for (; kk + (UNROLL - 1) < _PB_M; kk += UNROLL) {
#if UNROLL == 4
            acc += A[kk+0][i] * B[kk+0][j];
            acc += A[kk+1][i] * B[kk+1][j];
            acc += A[kk+2][i] * B[kk+2][j];
            acc += A[kk+3][i] * B[kk+3][j];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              acc += A[kk+uu][i] * B[kk+uu][j];
#endif
          }
          for (; kk < _PB_M; ++kk) acc += A[kk][i] * B[kk][j];
#else
          for (k = i + 1; k < _PB_M; ++k)
            acc += A[k][i] * B[k][j];
#endif
          B[i][j] = alpha * acc;
        } else {
          
          DATA_TYPE acc = B[i][j];
          for (k = i + 1; k < _PB_M; k += 2)
            acc += A[k][i] * B[k][j];
          acc *= SCALAR_VAL(2.0);
          B[i][j] = alpha * acc;
        }
      } else {
        
        if (condC && (m > N_TH)) {
          /* alternate hot */
          DATA_TYPE acc = B[i][j];
#if defined(LU) && (UNROLL > 1)
          int kk = i + 1;
          for (; kk + (UNROLL - 1) < _PB_M; kk += UNROLL) {
#if UNROLL == 4
            acc += A[kk+0][i] * B[kk+0][j];
            acc += A[kk+1][i] * B[kk+1][j];
            acc += A[kk+2][i] * B[kk+2][j];
            acc += A[kk+3][i] * B[kk+3][j];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              acc += A[kk+uu][i] * B[kk+uu][j];
#endif
          }
          for (; kk < _PB_M; ++kk) acc += A[kk][i] * B[kk][j];
#else
          for (k = i + 1; k < _PB_M; ++k)
            acc += A[k][i] * B[k][j];
#endif
          B[i][j] = alpha * acc;
        } else {
          /* cold path */
          DATA_TYPE acc = B[i][j];
          for (k = i + 1; k < _PB_M; k += 2)
            acc += A[k][i] * B[k][j];
          acc *= SCALAR_VAL(2.0);
          B[i][j] = alpha * acc;
        }
      }
#else
      /* baseline */
      DATA_TYPE acc = B[i][j];
      for (k = i + 1; k < _PB_M; k++)
        acc += A[k][i] * B[k][j];
      B[i][j] = alpha * acc;
#endif
    }
  }
#pragma endscop
}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trmm (m, n, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

