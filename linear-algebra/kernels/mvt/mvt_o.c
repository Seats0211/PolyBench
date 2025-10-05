/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i] = (DATA_TYPE) (i % n) / n;
      x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_mvt(int n,
        DATA_TYPE POLYBENCH_1D(x1,N,n),
        DATA_TYPE POLYBENCH_1D(x2,N,n),
        DATA_TYPE POLYBENCH_1D(y_1,N,n),
        DATA_TYPE POLYBENCH_1D(y_2,N,n),
        DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

#pragma scop
  /* First kernel: x1[i] += sum_j A[i][j] * y_1[j] */
  for (i = 0; i < _PB_N; i++) {
#if defined(SE)
    /* Shannon-style predicates (tunable) */
    const DATA_TYPE Y_TH = SCALAR_VAL(1e-6);   /* cheap threshold */
    const int J_TH = 64;                       /* problem-size threshold */
    int condA = (fabs(y_1[0]) > Y_TH);         /* cheap predicate */
    int condB = (i % 2 == 0);                  /* structural predicate */
    int condC = (((int)(A[i][0] * 1000.0)) & 1);/* pseudo-random-ish */

    DATA_TYPE acc1 = x1[i];

    if (condA) {
      if (condB || (condC && (_PB_N > J_TH))) {
        /* hot: full accumulation (support manual unroll) */
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
          acc1 += A[i][jj+0] * y_1[jj+0];
          acc1 += A[i][jj+1] * y_1[jj+1];
          acc1 += A[i][jj+2] * y_1[jj+2];
          acc1 += A[i][jj+3] * y_1[jj+3];
#else
          for (int uu=0; uu<UNROLL; ++uu) acc1 += A[i][jj+uu] * y_1[jj+uu];
#endif
        }
        for (; jj < _PB_N; ++jj) acc1 += A[i][jj] * y_1[jj];
#else
        for (j = 0; j < _PB_N; j++) acc1 += A[i][j] * y_1[j];
#endif
      } else {
        /* condA true but subguard false -> reduced-cost approximate (sample step=2) */
        for (j = 0; j < _PB_N; j += 2) acc1 += A[i][j] * y_1[j];
        acc1 *= SCALAR_VAL(2.0);
      }
    } else {
      /* condA false: alternate branch */
      if (condC && (_PB_N > J_TH)) {
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
          acc1 += A[i][jj+0] * y_1[jj+0];
          acc1 += A[i][jj+1] * y_1[jj+1];
          acc1 += A[i][jj+2] * y_1[jj+2];
          acc1 += A[i][jj+3] * y_1[jj+3];
#else
          for (int uu=0; uu<UNROLL; ++uu) acc1 += A[i][jj+uu] * y_1[jj+uu];
#endif
        }
        for (; jj < _PB_N; ++jj) acc1 += A[i][jj] * y_1[jj];
#else
        for (j = 0; j < _PB_N; j++) acc1 += A[i][j] * y_1[j];
#endif
      } else {
        /* cold: light approximate */
        for (j = 0; j < _PB_N; j += 2) acc1 += A[i][j] * y_1[j];
        acc1 *= SCALAR_VAL(2.0);
      }
    }

    x1[i] = acc1;

#else
    /* baseline exact */
    for (j = 0; j < _PB_N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
#endif
  }

  /* Second kernel: x2[i] += sum_j A[j][i] * y_2[j] */
  for (i = 0; i < _PB_N; i++) {
#if defined(SE)
    const DATA_TYPE Y2_TH = SCALAR_VAL(1e-6);
    const int J2_TH = 64;
    int condA2 = (fabs(y_2[0]) > Y2_TH);
    int condB2 = (i % 2 == 0);
    int condC2 = (((int)(A[0][i] * 1000.0)) & 1);

    DATA_TYPE acc2 = x2[i];

    if (condA2) {
      if (condB2 || (condC2 && (_PB_N > J2_TH))) {
        /* hot: full accumulation (support manual unroll over j) */
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
          acc2 += A[jj+0][i] * y_2[jj+0];
          acc2 += A[jj+1][i] * y_2[jj+1];
          acc2 += A[jj+2][i] * y_2[jj+2];
          acc2 += A[jj+3][i] * y_2[jj+3];
#else
          for (int uu=0; uu<UNROLL; ++uu) acc2 += A[jj+uu][i] * y_2[jj+uu];
#endif
        }
        for (; jj < _PB_N; ++jj) acc2 += A[jj][i] * y_2[jj];
#else
        for (j = 0; j < _PB_N; j++) acc2 += A[j][i] * y_2[j];
#endif
      } else {
        /* approximate */
        for (j = 0; j < _PB_N; j += 2) acc2 += A[j][i] * y_2[j];
        acc2 *= SCALAR_VAL(2.0);
      }
    } else {
      if (condC2 && (_PB_N > J2_TH)) {
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
          acc2 += A[jj+0][i] * y_2[jj+0];
          acc2 += A[jj+1][i] * y_2[jj+1];
          acc2 += A[jj+2][i] * y_2[jj+2];
          acc2 += A[jj+3][i] * y_2[jj+3];
#else
          for (int uu=0; uu<UNROLL; ++uu) acc2 += A[jj+uu][i] * y_2[jj+uu];
#endif
        }
        for (; jj < _PB_N; ++jj) acc2 += A[jj][i] * y_2[jj];
#else
        for (j = 0; j < _PB_N; j++) acc2 += A[j][i] * y_2[j];
#endif
      } else {
        for (j = 0; j < _PB_N; j += 2) acc2 += A[j][i] * y_2[j];
        acc2 *= SCALAR_VAL(2.0);
      }
    }

    x2[i] = acc2;

#else
    /* baseline exact */
    for (j = 0; j < _PB_N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
#endif
  }
#pragma endscop
}




int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_mvt (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}
