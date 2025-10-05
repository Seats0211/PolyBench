/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* cholesky.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++) {
    if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_cholesky(int n,
                     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    /* j < i : off-diagonal updates (we *may* apply SE here) */
    for (j = 0; j < i; j++) {
#if defined(SE)
      /* Shannon-style cheap predicates (tunable) */
      const DATA_TYPE A_TH = SCALAR_VAL(1e-6);   /* cheap threshold */
      const int  K_TH    = 64;                  /* problem-size threshold */

      int condA = (fabs(A[j][j]) > A_TH);               /* diagonal magnitude */
      int condB = ((i + j) % 2 == 0);                   /* structural predicate */
      int condC = (((int)(A[i][0] * 1000.0)) & 1);      /* pseudo-random-ish */

      DATA_TYPE acc = SCALAR_VAL(0.0);

      if (condA) {
        if (condB || (condC && (_PB_N > K_TH))) {
          /* hot path: full accurate accumulation */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < j; kk += UNROLL) {
#if UNROLL == 4
            acc += A[i][kk+0] * A[j][kk+0];
            acc += A[i][kk+1] * A[j][kk+1];
            acc += A[i][kk+2] * A[j][kk+2];
            acc += A[i][kk+3] * A[j][kk+3];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              acc += A[i][kk+uu] * A[j][kk+uu];
#endif
          }
          for (; kk < j; ++kk) acc += A[i][kk] * A[j][kk];
#else
          for (k = 0; k < j; k++)
            acc += A[i][k] * A[j][k];
#endif
        } else {
          /* condA true but subguard false -> reduced-cost approximate: sample every 2 */
          for (k = 0; k < j; k += 2)
            acc += A[i][k] * A[j][k];
          acc *= SCALAR_VAL(2.0); /* simple compensation */
        }
      } else {
        /* condA false: alternate grouping */
        if (condC && (_PB_N > K_TH)) {
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < j; kk += UNROLL) {
#if UNROLL == 4
            acc += A[i][kk+0] * A[j][kk+0];
            acc += A[i][kk+1] * A[j][kk+1];
            acc += A[i][kk+2] * A[j][kk+2];
            acc += A[i][kk+3] * A[j][kk+3];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              acc += A[i][kk+uu] * A[j][kk+uu];
#endif
          }
          for (; kk < j; ++kk) acc += A[i][kk] * A[j][kk];
#else
          for (k = 0; k < j; k++)
            acc += A[i][k] * A[j][k];
#endif
        } else {
          /* cold path: sampled approximate */
          for (k = 0; k < j; k += 2)
            acc += A[i][k] * A[j][k];
          acc *= SCALAR_VAL(2.0);
        }
      }

      /* apply subtraction and division (exact division by A[j][j]) */
      A[i][j] -= acc;
      A[i][j] /= A[j][j];

#else
      /* baseline exact code */
      for (k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[j][k];
      A[i][j] /= A[j][j];
#endif
    } /* end for j */

    /* i==i case: diagonal update. KEEP THIS EXACT for numerical stability. */
    {
      DATA_TYPE accd = SCALAR_VAL(0.0);
      for (k = 0; k < i; k++)
        accd += A[i][k] * A[i][k];
      A[i][i] -= accd;
      /* If numerical issues occur (negative due to rounding), clip to small positive */
      if (A[i][i] <= SCALAR_VAL(0.0)) {
        /* NOTE: clipping is a last-resort mitigation for experiment; report if happens */
        A[i][i] = SCALAR_VAL(1e-12);
      }
      A[i][i] = SQRT_FUN(A[i][i]);
    }
  } /* end for i */
#pragma endscop
}




int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky (n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}
