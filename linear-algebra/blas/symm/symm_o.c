/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_symm(int m, int n,
         DATA_TYPE alpha,
         DATA_TYPE beta,
         DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
         DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
         DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;
  DATA_TYPE temp2;

#pragma scop
  for (i = 0; i < _PB_M; i++) {
    for (j = 0; j < _PB_N; j++ ) {
#if defined(SE)
      /* ---------- Shannon-style predicates (tunable) ---------- */
      const DATA_TYPE ALPHA_TH = SCALAR_VAL(1e-3);
      const DATA_TYPE N_TH     = SCALAR_VAL(64.0);

      int condA = (fabs(alpha) > ALPHA_TH);        /* cheap predicate */
      int condB = ((i + j) % 2 == 0);              /* cheap structural */
      int condC = (((int)(A[i][0] * 1000.0)) & 1); /* pseudo-random bit */

      if (condA) {
        if (condB || (condC && (m > N_TH))) {
          /* hot path */
          temp2 = SCALAR_VAL(0.0);

#if defined(LU) && (UNROLL > 1)
          /* UNROLL == 4 */
          int kk = 0;
          for (; kk + (UNROLL - 1) < i; kk += UNROLL) {
#if UNROLL == 4
            
            /* k = kk+0 */
            C[kk+0][j] += alpha * B[i][j] * A[i][kk+0];
            temp2     += B[kk+0][j] * A[i][kk+0];
            /* k = kk+1 */
            C[kk+1][j] += alpha * B[i][j] * A[i][kk+1];
            temp2     += B[kk+1][j] * A[i][kk+1];
            /* k = kk+2 */
            C[kk+2][j] += alpha * B[i][j] * A[i][kk+2];
            temp2     += B[kk+2][j] * A[i][kk+2];
            /* k = kk+3 */
            C[kk+3][j] += alpha * B[i][j] * A[i][kk+3];
            temp2     += B[kk+3][j] * A[i][kk+3];
#else
            for (int uu = 0; uu < UNROLL; ++uu) {
              int kk2 = kk + uu;
              C[kk2][j] += alpha * B[i][j] * A[i][kk2];
              temp2     += B[kk2][j] * A[i][kk2];
            }
#endif
          }
          for (; kk < i; ++kk) {
            C[kk][j] += alpha * B[i][j] * A[i][kk];
            temp2     += B[kk][j] * A[i][kk];
          }
#else
          /* UNROLL == 1 */
          temp2 = SCALAR_VAL(0.0);
          for (k = 0; k < i; k++) {
            C[k][j] += alpha * B[i][j] * A[i][k];
            temp2     += B[k][j] * A[i][k];
          }
#endif
          
          C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
        } else {
          
          temp2 = SCALAR_VAL(0.0);
          
          for (k = 0; k < i; k += 2) {
            C[k][j] += alpha * B[i][j] * A[i][k];
            temp2   += B[k][j] * A[i][k];
          }
          
          temp2 *= SCALAR_VAL(2.0);
          
          for (k = 0; k < i; k += 2) {
            C[k][j] *= SCALAR_VAL(2.0);
          }
          
          C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
        }
      } else {
        
        if (condC && (m > N_TH)) {
          /* alternate hot path: full precise compute */
          temp2 = SCALAR_VAL(0.0);
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < i; kk += UNROLL) {
#if UNROLL == 4
            C[kk+0][j] += alpha * B[i][j] * A[i][kk+0];
            temp2     += B[kk+0][j] * A[i][kk+0];
            C[kk+1][j] += alpha * B[i][j] * A[i][kk+1];
            temp2     += B[kk+1][j] * A[i][kk+1];
            C[kk+2][j] += alpha * B[i][j] * A[i][kk+2];
            temp2     += B[kk+2][j] * A[i][kk+2];
            C[kk+3][j] += alpha * B[i][j] * A[i][kk+3];
            temp2     += B[kk+3][j] * A[i][kk+3];
#else
            for (int uu = 0; uu < UNROLL; ++uu) {
              int kk2 = kk + uu;
              C[kk2][j] += alpha * B[i][j] * A[i][kk2];
              temp2     += B[kk2][j] * A[i][kk2];
            }
#endif
          }
          for (; kk < i; ++kk) {
            C[kk][j] += alpha * B[i][j] * A[i][kk];
            temp2     += B[kk][j] * A[i][kk];
          }
#else
          temp2 = SCALAR_VAL(0.0);
          for (k = 0; k < i; k++) {
            C[k][j] += alpha * B[i][j] * A[i][k];
            temp2     += B[k][j] * A[i][k];
          }
#endif
          C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
        } else {
          /* cold path: lighter approximate (sampling) */
          temp2 = SCALAR_VAL(0.0);
          for (k = 0; k < i; k += 2) {
            C[k][j] += alpha * B[i][j] * A[i][k];
            temp2   += B[k][j] * A[i][k];
          }
          temp2 *= SCALAR_VAL(2.0);
          for (k = 0; k < i; k += 2) {
            C[k][j] *= SCALAR_VAL(2.0);
          }
          C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
        }
      }
#else
      /* ---------- baseline ---------- */
      temp2 = SCALAR_VAL(0.0);
      for (k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2     += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
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
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_symm (m, n,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

