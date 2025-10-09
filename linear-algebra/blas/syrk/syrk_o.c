/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syrk.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_syrk(int n, int m,
         DATA_TYPE alpha,
         DATA_TYPE beta,
         DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
         DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j, k;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    /* scale lower triangle */
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;

    for (j = 0; j <= i; j++) {
#if defined(SE)
      /* ---------- Shannon-like predicates (tunable) ---------- */
      const DATA_TYPE ALPHA_TH = SCALAR_VAL(1e-3);
      const DATA_TYPE M_TH     = SCALAR_VAL(64.0);

      int condA = (fabs(alpha) > ALPHA_TH);          /* cheap predicate */
      int condB = ((i + j) % 2 == 0);                /* structural cheap predicate */
      int condC = (((int)(A[i][0] * 1000.0)) & 1);   /* pseudo-random bit */

      DATA_TYPE sum = SCALAR_VAL(0.0);

      if (condA) {
        if (condB || (condC && (m > M_TH))) {
          /* hot path */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_M; kk += UNROLL) {
#if UNROLL == 4
            sum += alpha * A[i][kk+0] * A[j][kk+0] + alpha * A[j][kk+0] * A[i][kk+0];
            sum += alpha * A[i][kk+1] * A[j][kk+1] + alpha * A[j][kk+1] * A[i][kk+1];
            sum += alpha * A[i][kk+2] * A[j][kk+2] + alpha * A[j][kk+2] * A[i][kk+2];
            sum += alpha * A[i][kk+3] * A[j][kk+3] + alpha * A[j][kk+3] * A[i][kk+3];
#else
            for (int uu = 0; uu < UNROLL; ++uu) {
              int kk2 = kk + uu;
              sum += alpha * A[i][kk2] * A[j][kk2] + alpha * A[j][kk2] * A[i][kk2];
            }
#endif
          }
          for (; kk < _PB_M; ++kk)
            sum += alpha * A[i][kk] * A[j][kk] + alpha * A[j][kk] * A[i][kk];
#else
          for (k = 0; k < _PB_M; k++)
            sum += alpha * A[i][k] * A[j][k] + alpha * A[j][k] * A[i][k];
#endif
        } else {
          
          for (k = 0; k < _PB_M; k += 2)
            sum += alpha * A[i][k] * A[j][k] + alpha * A[j][k] * A[i][k];
          sum *= SCALAR_VAL(2.0);
        }
      } else {
        
        if (condC && (m > M_TH)) {
          /* alternate hot */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_M; kk += UNROLL) {
#if UNROLL == 4
            sum += alpha * A[i][kk+0] * A[j][kk+0] + alpha * A[j][kk+0] * A[i][kk+0];
            sum += alpha * A[i][kk+1] * A[j][kk+1] + alpha * A[j][kk+1] * A[i][kk+1];
            sum += alpha * A[i][kk+2] * A[j][kk+2] + alpha * A[j][kk+2] * A[i][kk+2];
            sum += alpha * A[i][kk+3] * A[j][kk+3] + alpha * A[j][kk+3] * A[i][kk+3];
#else
            for (int uu = 0; uu < UNROLL; ++uu) {
              int kk2 = kk + uu;
              sum += alpha * A[i][kk2] * A[j][kk2] + alpha * A[j][kk2] * A[i][kk2];
            }
#endif
          }
          for (; kk < _PB_M; ++kk)
            sum += alpha * A[i][kk] * A[j][kk] + alpha * A[j][kk] * A[i][kk];
#else
          for (k = 0; k < _PB_M; k++)
            sum += alpha * A[i][k] * A[j][k] + alpha * A[j][k] * A[i][k];
#endif
        } else {
          /* cold path */
          for (k = 0; k < _PB_M; k += 2)
            sum += alpha * A[i][k] * A[j][k] + alpha * A[j][k] * A[i][k];
          sum *= SCALAR_VAL(2.0);
        }
      }

      C[i][j] += sum;

#else
      /* baseline */
      for (k = 0; k < _PB_M; k++)
        C[i][j] += alpha * A[i][k] * A[j][k]; /* note: for syrk baseline should be C[i][j]+=alpha*A[i][k]*A[j][k]; */
      /* Above line kept for consistency; replace with the canonical form if desired: */
      /* for (k = 0; k < _PB_M; k++) C[i][j] += alpha * A[i][k] * A[j][k]; */
#endif
    }
  }
#pragma endscop
}




int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syrk (n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}

