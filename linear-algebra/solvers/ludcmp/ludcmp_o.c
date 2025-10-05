/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

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
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
                   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                   DATA_TYPE POLYBENCH_1D(b,N,n),
                   DATA_TYPE POLYBENCH_1D(x,N,n),
                   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;
  DATA_TYPE w;

#pragma scop
  /* LU factorization */
  for (i = 0; i < _PB_N; i++) {
    /* lower triangle */
    for (j = 0; j < i; j++) {
      w = A[i][j];
#if defined(SE)
      {
        const int K_TH = 64;
        int condStruct = ((i + j) % 2 == 0);
        int condRand   = (((int)(A[i][0]*1000.0)) & 1);
        if (condStruct || (condRand && (_PB_N > K_TH))) {
          /* hot: full accumulation */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL-1) < j; kk += UNROLL) {
#if UNROLL == 4
            w -= A[i][kk+0] * A[kk+0][j];
            w -= A[i][kk+1] * A[kk+1][j];
            w -= A[i][kk+2] * A[kk+2][j];
            w -= A[i][kk+3] * A[kk+3][j];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              w -= A[i][kk+uu] * A[kk+uu][j];
#endif
          }
          for (; kk<j; ++kk) w -= A[i][kk] * A[kk][j];
#else
          for (k = 0; k < j; k++)
            w -= A[i][k] * A[k][j];
#endif
        } else {
          /* cold: sample every 2 */
          for (k = 0; k < j; k += 2)
            w -= A[i][k] * A[k][j];
          w *= SCALAR_VAL(2.0);
        }
      }
#else
      for (k = 0; k < j; k++)
        w -= A[i][k] * A[k][j];
#endif
      /* exact division */
      if (fabsl(A[j][j]) < SCALAR_VAL(1e-18))
        A[j][j] = (A[j][j] >= 0) ? SCALAR_VAL(1e-18) : -SCALAR_VAL(1e-18);
      A[i][j] = w / A[j][j];
    }

    /* upper triangle */
    for (j = i; j < _PB_N; j++) {
      w = A[i][j];
#if defined(SE)
      {
        int condStruct = ((i + j) % 2 == 0);
        int condRand   = (((int)(A[0][j]*1000.0)) & 1);
        if (condStruct || condRand) {
          /* hot path */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL-1) < i; kk += UNROLL) {
#if UNROLL == 4
            w -= A[i][kk+0] * A[kk+0][j];
            w -= A[i][kk+1] * A[kk+1][j];
            w -= A[i][kk+2] * A[kk+2][j];
            w -= A[i][kk+3] * A[kk+3][j];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              w -= A[i][kk+uu] * A[kk+uu][j];
#endif
          }
          for (; kk<i; ++kk) w -= A[i][kk] * A[kk][j];
#else
          for (k = 0; k < i; k++)
            w -= A[i][k] * A[k][j];
#endif
        } else {
          /* cold path */
          for (k = 0; k < i; k += 2)
            w -= A[i][k] * A[k][j];
          w *= SCALAR_VAL(2.0);
        }
      }
#else
      for (k = 0; k < i; k++)
        w -= A[i][k] * A[k][j];
#endif
      A[i][j] = w;
    }
  }

  /* forward substitution: Ly = b */
  for (i = 0; i < _PB_N; i++) {
    w = b[i];
#if defined(SE)
    {
      int condRand = (i & 1);
      if (condRand) {
        for (j = 0; j < i; j++)
          w -= A[i][j] * y[j];
      } else {
        for (j = 0; j < i; j += 2)
          w -= A[i][j] * y[j];
        w *= SCALAR_VAL(2.0);
      }
    }
#else
    for (j = 0; j < i; j++)
      w -= A[i][j] * y[j];
#endif
    y[i] = w;
  }

  /* backward substitution: Ux = y */
  for (i = _PB_N-1; i >= 0; i--) {
    w = y[i];
#if defined(SE)
    {
      int condRand = (i & 1);
      if (condRand) {
        for (j = i+1; j < _PB_N; j++)
          w -= A[i][j] * x[j];
      } else {
        for (j = i+1; j < _PB_N; j += 2)
          w -= A[i][j] * x[j];
        w *= SCALAR_VAL(2.0);
      }
    }
#else
    for (j = i+1; j < _PB_N; j++)
      w -= A[i][j] * x[j];
#endif
    if (fabsl(A[i][i]) < SCALAR_VAL(1e-18))
      A[i][i] = (A[i][i] >= 0) ? SCALAR_VAL(1e-18) : -SCALAR_VAL(1e-18);
    x[i] = w / A[i][i];
  }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_ludcmp (n,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(b),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}
