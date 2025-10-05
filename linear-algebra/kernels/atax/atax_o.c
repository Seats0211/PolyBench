/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_atax(int m, int n,
                 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                 DATA_TYPE POLYBENCH_1D(x,N,n),
                 DATA_TYPE POLYBENCH_1D(y,N,n),
                 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

#pragma scop
  /* initialize y */
  for (j = 0; j < _PB_N; j++)
    y[j] = SCALAR_VAL(0.0);

  /* main loops */
  for (i = 0; i < _PB_M; i++)
  {
#if defined(SE)
    /* Shannon-style predicates (cheap & tunable) */
    const DATA_TYPE X_TH = SCALAR_VAL(1e-3);
    const DATA_TYPE N_TH = SCALAR_VAL(64.0); /* tune per problem size */

    int condA = (fabs(x[0]) > X_TH);           /* cheap predicate */
    int condB = (i % 2 == 0);                  /* structural predicate */
    int condC = (((int)(A[i][0] * 1000.0)) & 1);/* pseudo-random-ish */

    /* 1) compute tmp[i] = sum_j A[i][j] * x[j]  */
    DATA_TYPE tmp_acc = SCALAR_VAL(0.0);
    if (condA) {
      if (condB || (condC && (n > N_TH))) {
        /* hot path: full accumulation */
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
          tmp_acc += A[i][jj+0] * x[jj+0];
          tmp_acc += A[i][jj+1] * x[jj+1];
          tmp_acc += A[i][jj+2] * x[jj+2];
          tmp_acc += A[i][jj+3] * x[jj+3];
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            tmp_acc += A[i][jj+uu] * x[jj+uu];
#endif
        }
        for (; jj < _PB_N; ++jj) tmp_acc += A[i][jj] * x[jj];
#else
        for (j = 0; j < _PB_N; ++j)
          tmp_acc += A[i][j] * x[j];
#endif
      } else {
        /* condA true but subguard false -> reduced-cost approximate: sample every 2 */
        for (j = 0; j < _PB_N; j += 2)
          tmp_acc += A[i][j] * x[j];
        tmp_acc *= SCALAR_VAL(2.0); /* simple compensation */
      }
    } else {
      /* condA false: alternate branching */
      if (condC && (n > N_TH)) {
        /* alternate hot: full accumulation */
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
          tmp_acc += A[i][jj+0] * x[jj+0];
          tmp_acc += A[i][jj+1] * x[jj+1];
          tmp_acc += A[i][jj+2] * x[jj+2];
          tmp_acc += A[i][jj+3] * x[jj+3];
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            tmp_acc += A[i][jj+uu] * x[jj+uu];
#endif
        }
        for (; jj < _PB_N; ++jj) tmp_acc += A[i][jj] * x[jj];
#else
        for (j = 0; j < _PB_N; ++j)
          tmp_acc += A[i][j] * x[j];
#endif
      } else {
        /* cold path: light approximation */
        for (j = 0; j < _PB_N; j += 2)
          tmp_acc += A[i][j] * x[j];
        tmp_acc *= SCALAR_VAL(2.0);
      }
    }
    tmp[i] = tmp_acc;

    /* 2) update y: y[j] += A[i][j] * tmp[i]  */
    if (condA) {
      if (condB || (condC && (n > N_TH))) {
        /* hot path: full update */
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
          y[jj+0] += A[i][jj+0] * tmp_acc;
          y[jj+1] += A[i][jj+1] * tmp_acc;
          y[jj+2] += A[i][jj+2] * tmp_acc;
          y[jj+3] += A[i][jj+3] * tmp_acc;
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            y[jj+uu] += A[i][jj+uu] * tmp_acc;
#endif
        }
        for (; jj < _PB_N; ++jj) y[jj] += A[i][jj] * tmp_acc;
#else
        for (j = 0; j < _PB_N; ++j)
          y[j] += A[i][j] * tmp_acc;
#endif
      } else {
        /* reduced-cost update: sample every 2 and scale */
        for (j = 0; j < _PB_N; j += 2)
          y[j] += A[i][j] * tmp_acc;
        /* rough compensation for skipped indices */
        for (j = 0; j < _PB_N; j += 2)
          y[j] *= SCALAR_VAL(2.0);
      }
    } else {
      if (condC && (n > N_TH)) {
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
          y[jj+0] += A[i][jj+0] * tmp_acc;
          y[jj+1] += A[i][jj+1] * tmp_acc;
          y[jj+2] += A[i][jj+2] * tmp_acc;
          y[jj+3] += A[i][jj+3] * tmp_acc;
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            y[jj+uu] += A[i][jj+uu] * tmp_acc;
#endif
        }
        for (; jj < _PB_N; ++jj) y[jj] += A[i][jj] * tmp_acc;
#else
        for (j = 0; j < _PB_N; ++j)
          y[j] += A[i][j] * tmp_acc;
#endif
      } else {
        /* cold path: sampled update */
        for (j = 0; j < _PB_N; j += 2)
          y[j] += A[i][j] * tmp_acc;
        for (j = 0; j < _PB_N; j += 2)
          y[j] *= SCALAR_VAL(2.0);
      }
    }

#else
    /* baseline: exact compute */
    tmp[i] = SCALAR_VAL(0.0);
    for (j = 0; j < _PB_N; j++)
      tmp[i] += A[i][j] * x[j];
    for (j = 0; j < _PB_N; j++)
      y[j] += A[i][j] * tmp[i];
#endif
  }
#pragma endscop
}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}
