/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

#pragma scop

  /* Strategy B tuning parameters: */
  const DATA_TYPE U1_TH = SCALAR_VAL(1e-6);
  const int SAMPLE_STEP = 2;

  /* 1) Rank-1 updates: A[i][j] += u1[i]*v1[j] + u2[i]*v2[j]
     This update is usually dense with few branches - 
	 we retain the exact update but provide optional manual unrolling (LU) */
#if defined(SE)
  /* We keep this exact but allow optional unrolling to show LU effect */
#endif

#if defined(LU) && (UNROLL > 1)
  for (i = 0; i < _PB_N; ++i) {
    int jj = 0;
    for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
      A[i][jj  ] = A[i][jj  ] + u1[i]*v1[jj  ] + u2[i]*v2[jj  ];
      A[i][jj+1] = A[i][jj+1] + u1[i]*v1[jj+1] + u2[i]*v2[jj+1];
      A[i][jj+2] = A[i][jj+2] + u1[i]*v1[jj+2] + u2[i]*v2[jj+2];
      A[i][jj+3] = A[i][jj+3] + u1[i]*v1[jj+3] + u2[i]*v2[jj+3];
#else
      for (int uu=0; uu<UNROLL; ++uu)
        A[i][jj+uu] = A[i][jj+uu] + u1[i]*v1[jj+uu] + u2[i]*v2[jj+uu];
#endif
    }
    for (; jj < _PB_N; ++jj) A[i][jj] = A[i][jj] + u1[i]*v1[jj] + u2[i]*v2[jj];
  }
#else
  for (i = 0; i < _PB_N; ++i)
    for (j = 0; j < _PB_N; ++j)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
#endif

#if defined(SE)
  
  const DATA_TYPE U_TH = SCALAR_VAL(1e-3);
#endif

  for (i = 0; i < _PB_N; ++i) {
#if defined(SE)
    
    int condA = (fabs(u1[i]) > U_TH || fabs(u2[i]) > U_TH);
    if (condA) {
      /* hot path */
#if defined(LU) && (UNROLL > 1)
      DATA_TYPE acc = SCALAR_VAL(0.0);
      int jj = 0;
      for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
        acc += beta * A[jj  ][i] * y[jj  ];
        acc += beta * A[jj+1][i] * y[jj+1];
        acc += beta * A[jj+2][i] * y[jj+2];
        acc += beta * A[jj+3][i] * y[jj+3];
#else
        for (int uu=0; uu<UNROLL; ++uu) acc += beta * A[jj+uu][i] * y[jj+uu];
#endif
      }
      for (; jj < _PB_N; ++jj) acc += beta * A[jj][i] * y[jj];
      x[i] += acc;
#else
      for (j = 0; j < _PB_N; ++j) x[i] += beta * A[j][i] * y[j];
#endif
    } else {
      /* cold path */
      DATA_TYPE acc = SCALAR_VAL(0.0);
      int cnt = 0;
      for (j = 0; j < _PB_N; j += SAMPLE_STEP) {
        acc += beta * A[j][i] * y[j];
        cnt++;
      }
    
      x[i] += acc * (DATA_TYPE)SAMPLE_STEP;
    }
#else
    /* baseline exact */
    for (j = 0; j < _PB_N; ++j) x[i] += beta * A[j][i] * y[j];
#endif
  }

  /* 3) x += z (exact, cheap) */
  for (i = 0; i < _PB_N; ++i)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; ++i) {
#if defined(SE)
    int condA = (fabs(x[i]) > U_TH);
    if (condA) {
#if defined(LU) && (UNROLL > 1)
      DATA_TYPE acc = SCALAR_VAL(0.0);
      int jj = 0;
      for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
        acc += alpha * A[i][jj  ] * x[jj  ];
        acc += alpha * A[i][jj+1] * x[jj+1];
        acc += alpha * A[i][jj+2] * x[jj+2];
        acc += alpha * A[i][jj+3] * x[jj+3];
#else
        for (int uu=0; uu<UNROLL; ++uu) acc += alpha * A[i][jj+uu] * x[jj+uu];
#endif
      }
      for (; jj < _PB_N; ++jj) acc += alpha * A[i][jj] * x[jj];
      w[i] += acc;
#else
      for (j = 0; j < _PB_N; ++j) w[i] += alpha * A[i][j] * x[j];
#endif
    } else {
      /* cold path */
      DATA_TYPE acc = SCALAR_VAL(0.0);
      int cnt = 0;
      for (j = 0; j < _PB_N; j += SAMPLE_STEP) {
        acc += alpha * A[i][j] * x[j];
        cnt++;
      }
      w[i] += acc * (DATA_TYPE)SAMPLE_STEP;
    }
#else
    for (j = 0; j < _PB_N; ++j) w[i] += alpha * A[i][j] * x[j];
#endif
  }

#pragma endscop
}




int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemver (n, alpha, beta,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(u1),
		 POLYBENCH_ARRAY(v1),
		 POLYBENCH_ARRAY(u2),
		 POLYBENCH_ARRAY(v2),
		 POLYBENCH_ARRAY(w),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y),
		 POLYBENCH_ARRAY(z));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}

