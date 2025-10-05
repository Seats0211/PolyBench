/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gramschmidt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* QR Decomposition with Modified Gram Schmidt:
 http://www.inf.ethz.ch/personal/gander/ */
static
void kernel_gramschmidt(int m, int n,
            DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
            DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
            DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;

  DATA_TYPE nrm;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      /* 1) norm = ||A[:,k]||_2  (keep this exact for stability) */
      nrm = SCALAR_VAL(0.0);
#if defined(LU) && (UNROLL > 1)
      {
        int ii = 0;
        for (; ii + (UNROLL-1) < _PB_M; ii += UNROLL) {
#if UNROLL == 4
          nrm += A[ii+0][k] * A[ii+0][k];
          nrm += A[ii+1][k] * A[ii+1][k];
          nrm += A[ii+2][k] * A[ii+2][k];
          nrm += A[ii+3][k] * A[ii+3][k];
#else
          for (int uu=0; uu<UNROLL; ++uu)
            nrm += A[ii+uu][k] * A[ii+uu][k];
#endif
        }
        for (; ii < _PB_M; ++ii) nrm += A[ii][k] * A[ii][k];
      }
#else
      for (i = 0; i < _PB_M; i++)
        nrm += A[i][k] * A[i][k];
#endif
      R[k][k] = SQRT_FUN(nrm);

      /* Avoid division by zero or extremely small diagonal */
      if (R[k][k] == SCALAR_VAL(0.0)) R[k][k] = SCALAR_VAL(1e-12);

      /* 2) Q[:,k] = A[:,k] / R[k][k]  (keep exact) */
#if defined(LU) && (UNROLL > 1)
      {
        int ii = 0;
        for (; ii + (UNROLL-1) < _PB_M; ii += UNROLL) {
#if UNROLL == 4
          Q[ii+0][k] = A[ii+0][k] / R[k][k];
          Q[ii+1][k] = A[ii+1][k] / R[k][k];
          Q[ii+2][k] = A[ii+2][k] / R[k][k];
          Q[ii+3][k] = A[ii+3][k] / R[k][k];
#else
          for (int uu=0; uu<UNROLL; ++uu)
            Q[ii+uu][k] = A[ii+uu][k] / R[k][k];
#endif
        }
        for (; ii < _PB_M; ++ii) Q[ii][k] = A[ii][k] / R[k][k];
      }
#else
      for (i = 0; i < _PB_M; i++)
        Q[i][k] = A[i][k] / R[k][k];
#endif

      /* 3) For j>k compute R[k][j] = Q[:,k]^T * A[:,j] and update A[:,j] -= Q[:,k] * R[k][j]
         We can apply SE-style optimization on the inner-products and updates.
         Note: care must be taken —— approximations affect orthogonality and numerical stability.
      */
      for (j = k + 1; j < _PB_N; j++)
  {
#if defined(SE)
    /* Shannon-style predicates (cheap & tunable) */
    const DATA_TYPE QCOL_TH = SCALAR_VAL(1e-6);
    const int M_TH = 64; /* threshold on _PB_M to decide whether sampling helps */

    int condA = (fabs(Q[0][k]) > QCOL_TH);         /* cheap predicate on Q col */
    int condB = ((k + j) % 2 == 0);               /* structural */
    int condC = (((int)(A[0][j] * 1000.0)) & 1);   /* pseudo-random */

    DATA_TYPE dot = SCALAR_VAL(0.0);

    if (condA) {
      if (condB || (condC && (_PB_M > M_TH))) {
        /* hot path: full accurate inner product */
  #if defined(LU) && (UNROLL > 1)
        int ii = 0;
        for (; ii + (UNROLL-1) < _PB_M; ii += UNROLL) {
  #if UNROLL == 4
          dot += Q[ii+0][k] * A[ii+0][j];
          dot += Q[ii+1][k] * A[ii+1][j];
          dot += Q[ii+2][k] * A[ii+2][j];
          dot += Q[ii+3][k] * A[ii+3][j];
  #else
          for (int uu=0; uu<UNROLL; ++uu)
            dot += Q[ii+uu][k] * A[ii+uu][j];
  #endif
        }
        for (; ii < _PB_M; ++ii) dot += Q[ii][k] * A[ii][j];
  #else
        for (i = 0; i < _PB_M; i++)
          dot += Q[i][k] * A[i][j];
  #endif
      } else {
        /* condA true but subguard false -> reduced-cost approximate: sample every 2 */
        for (i = 0; i < _PB_M; i += 2)
          dot += Q[i][k] * A[i][j];
        dot *= SCALAR_VAL(2.0); /* simple compensation */
      }
    } else {
      /* condA false: alternate grouping */
      if (condC && (_PB_M > M_TH)) {
  #if defined(LU) && (UNROLL > 1)
        int ii = 0;
        for (; ii + (UNROLL-1) < _PB_M; ii += UNROLL) {
  #if UNROLL == 4
          dot += Q[ii+0][k] * A[ii+0][j];
          dot += Q[ii+1][k] * A[ii+1][j];
          dot += Q[ii+2][k] * A[ii+2][j];
          dot += Q[ii+3][k] * A[ii+3][j];
  #else
          for (int uu=0; uu<UNROLL; ++uu)
            dot += Q[ii+uu][k] * A[ii+uu][j];
  #endif
        }
        for (; ii < _PB_M; ++ii) dot += Q[ii][k] * A[ii][j];
  #else
        for (i = 0; i < _PB_M; i++)
          dot += Q[i][k] * A[i][j];
  #endif
      } else {
        /* cold path: sampled approximate */
        for (i = 0; i < _PB_M; i += 2)
          dot += Q[i][k] * A[i][j];
        dot *= SCALAR_VAL(2.0);
      }
    }

    R[k][j] = dot;

    /* Update A[:,j] -= Q[:,k] * R[k][j]  (apply same hot/cold split for updates) */
    if (condA) {
      if (condB || (condC && (_PB_M > M_TH))) {
  #if defined(LU) && (UNROLL > 1)
        int ii = 0;
        for (; ii + (UNROLL-1) < _PB_M; ii += UNROLL) {
  #if UNROLL == 4
          A[ii+0][j] -= Q[ii+0][k] * dot;
          A[ii+1][j] -= Q[ii+1][k] * dot;
          A[ii+2][j] -= Q[ii+2][k] * dot;
          A[ii+3][j] -= Q[ii+3][k] * dot;
  #else
          for (int uu=0; uu<UNROLL; ++uu)
            A[ii+uu][j] -= Q[ii+uu][k] * dot;
  #endif
        }
        for (; ii < _PB_M; ++ii) A[ii][j] -= Q[ii][k] * dot;
  #else
        for (i = 0; i < _PB_M; i++)
          A[i][j] -= Q[i][k] * dot;
  #endif
      } else {
        /* approximate update: sample every 2 and scale (rough) */
        for (i = 0; i < _PB_M; i += 2)
          A[i][j] -= Q[i][k] * dot;
        for (i = 0; i < _PB_M; i += 2)
          A[i][j] *= SCALAR_VAL(2.0); /* crude compensation */
      }
    } else {
      if (condC && (_PB_M > M_TH)) {
  #if defined(LU) && (UNROLL > 1)
        int ii = 0;
        for (; ii + (UNROLL-1) < _PB_M; ii += UNROLL) {
  #if UNROLL == 4
          A[ii+0][j] -= Q[ii+0][k] * dot;
          A[ii+1][j] -= Q[ii+1][k] * dot;
          A[ii+2][j] -= Q[ii+2][k] * dot;
          A[ii+3][j] -= Q[ii+3][k] * dot;
  #else
          for (int uu=0; uu<UNROLL; ++uu)
            A[ii+uu][j] -= Q[ii+uu][k] * dot;
  #endif
        }
        for (; ii < _PB_M; ++ii) A[ii][j] -= Q[ii][k] * dot;
  #else
        for (i = 0; i < _PB_M; i++)
          A[i][j] -= Q[i][k] * dot;
  #endif
      } else {
        for (i = 0; i < _PB_M; i += 2)
          A[i][j] -= Q[i][k] * dot;
        for (i = 0; i < _PB_M; i += 2)
          A[i][j] *= SCALAR_VAL(2.0);
      }
    }

#else
    /* baseline: exact inner product and update */
    R[k][j] = SCALAR_VAL(0.0);
  #if defined(LU) && (UNROLL > 1)
    {
      int ii = 0;
      for (; ii + (UNROLL-1) < _PB_M; ii += UNROLL) {
  #if UNROLL == 4
        R[k][j] += Q[ii+0][k] * A[ii+0][j];
        R[k][j] += Q[ii+1][k] * A[ii+1][j];
        R[k][j] += Q[ii+2][k] * A[ii+2][j];
        R[k][j] += Q[ii+3][k] * A[ii+3][j];
  #else
        for (int uu=0; uu<UNROLL; ++uu)
          R[k][j] += Q[ii+uu][k] * A[ii+uu][j];
  #endif
      }
      for (; ii < _PB_M; ++ii) R[k][j] += Q[ii][k] * A[ii][j];
    }
  #else
    for (i = 0; i < _PB_M; i++)
      R[k][j] += Q[i][k] * A[i][j];
  #endif

  #if defined(LU) && (UNROLL > 1)
    {
      int ii = 0;
      for (; ii + (UNROLL-1) < _PB_M; ii += UNROLL) {
  #if UNROLL == 4
        A[ii+0][j] -= Q[ii+0][k] * R[k][j];
        A[ii+1][j] -= Q[ii+1][k] * R[k][j];
        A[ii+2][j] -= Q[ii+2][k] * R[k][j];
        A[ii+3][j] -= Q[ii+3][k] * R[k][j];
  #else
        for (int uu=0; uu<UNROLL; ++uu)
          A[ii+uu][j] -= Q[ii+uu][k] * R[k][j];
  #endif
      }
      for (; ii < _PB_M; ++ii) A[ii][j] -= Q[ii][k] * R[k][j];
    }
  #else
    for (i = 0; i < _PB_M; i++)
      A[i][j] -= Q[i][k] * R[k][j];
  #endif
#endif
  } /* end for j */
    }
#pragma endscop

}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt (m, n,
		      POLYBENCH_ARRAY(A),
		      POLYBENCH_ARRAY(R),
		      POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}
