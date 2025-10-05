/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* lu.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lu.h"


/* Array initialization. */
static
void init_array (int n,
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
  /* not necessary for LU, but using same code as cholesky */
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
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_lu(int n,
               DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    /* lower triangle: compute A[i][j] for j < i (with division by A[j][j]) */
    for (j = 0; j < i; j++) {
#if defined(SE)
      /* Shannon-style predicates (tunable) */
      const DATA_TYPE DIAG_TH = SCALAR_VAL(1e-8);  /* guard on pivot magnitude */
      const int K_TH = 64;                        /* size threshold for sampling */
      const DATA_TYPE SUM_TH = SCALAR_VAL(1e-3);  /* secondary guard on sum */

      int condPivot = (fabsl(A[j][j]) > DIAG_TH);           /* pivot safe? */
      int condStruct = ((i + j) % 2 == 0);                  /* structural predicate */
      int condRand = (((int)(A[i][0] * 1000.0)) & 1);      /* pseudo-random bit */

      DATA_TYPE sum = SCALAR_VAL(0.0);

      if (condPivot) {
        if (condStruct || (condRand && (_PB_N > K_TH))) {
          /* hot path: full accurate accumulation */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < j; kk += UNROLL) {
#if UNROLL == 4
            sum += A[i][kk+0] * A[kk+0][j];
            sum += A[i][kk+1] * A[kk+1][j];
            sum += A[i][kk+2] * A[kk+2][j];
            sum += A[i][kk+3] * A[kk+3][j];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              sum += A[i][kk+uu] * A[kk+uu][j];
#endif
          }
          for (; kk < j; ++kk) sum += A[i][kk] * A[kk][j];
#else
          for (k = 0; k < j; k++)
            sum += A[i][k] * A[k][j];
#endif
        } else {
          /* condPivot true but subguard false -> approximate: sample every 2 */
          for (k = 0; k < j; k += 2)
            sum += A[i][k] * A[k][j];
          sum *= SCALAR_VAL(2.0); /* crude compensation */
        }
      } else {
        /* pivot small: be conservative — use full accumulation to avoid instability */
#if defined(LU) && (UNROLL > 1)
        int kk = 0;
        for (; kk + (UNROLL - 1) < j; kk += UNROLL) {
#if UNROLL == 4
          sum += A[i][kk+0] * A[kk+0][j];
          sum += A[i][kk+1] * A[kk+1][j];
          sum += A[i][kk+2] * A[kk+2][j];
          sum += A[i][kk+3] * A[kk+3][j];
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            sum += A[i][kk+uu] * A[kk+uu][j];
#endif
        }
        for (; kk < j; ++kk) sum += A[i][kk] * A[kk][j];
#else
        for (k = 0; k < j; k++)
          sum += A[i][k] * A[k][j];
#endif
      }

      /* subtract accumulated sum (exact subtraction) */
      A[i][j] -= sum;

      /* exact division by pivot (do not approximate) */
      /* guard tiny pivot to avoid division by near-zero (report if happens) */
      if (fabsl(A[j][j]) < SCALAR_VAL(1e-18)) {
        /* clamp pivot to tiny value to avoid blow-up; record in experiments if occurs */
        A[j][j] = (A[j][j] >= 0) ? SCALAR_VAL(1e-18) : -SCALAR_VAL(1e-18);
      }
      A[i][j] /= A[j][j];

#else
      /* baseline exact */
      for (k = 0; k < j; k++)
        A[i][j] -= A[i][k] * A[k][j];
      if (fabsl(A[j][j]) < SCALAR_VAL(1e-18)) {
        A[j][j] = (A[j][j] >= 0) ? SCALAR_VAL(1e-18) : -SCALAR_VAL(1e-18);
      }
      A[i][j] /= A[j][j];
#endif
    }

    /* upper triangle: compute A[i][j] for j >= i (no division) */
    for (j = i; j < _PB_N; j++) {
#if defined(SE)
      const DATA_TYPE SUM_TH_U = SCALAR_VAL(1e-3);
      const int K_TH_U = 64;
      int condStructU = ((i + j) % 2 == 0);
      int condRandU = (((int)(A[0][j] * 1000.0)) & 1);

      DATA_TYPE sumu = SCALAR_VAL(0.0);

      if (condStructU || (condRandU && (_PB_N > K_TH_U))) {
        /* hot: full accumulation */
#if defined(LU) && (UNROLL > 1)
        int kk = 0;
        for (; kk + (UNROLL - 1) < i; kk += UNROLL) {
#if UNROLL == 4
          sumu += A[i][kk+0] * A[kk+0][j];
          sumu += A[i][kk+1] * A[kk+1][j];
          sumu += A[i][kk+2] * A[kk+2][j];
          sumu += A[i][kk+3] * A[kk+3][j];
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            sumu += A[i][kk+uu] * A[kk+uu][j];
#endif
        }
        for (; kk < i; ++kk) sumu += A[i][kk] * A[kk][j];
#else
        for (k = 0; k < i; k++)
          sumu += A[i][k] * A[k][j];
#endif
      } else {
        /* cold approximate */
        for (k = 0; k < i; k += 2)
          sumu += A[i][k] * A[k][j];
        sumu *= SCALAR_VAL(2.0);
      }

      A[i][j] -= sumu;
#else
      /* baseline exact */
      for (k = 0; k < i; k++)
        A[i][j] -= A[i][k] * A[k][j];
#endif
    }
  }
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
  kernel_lu (n, POLYBENCH_ARRAY(A));

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
