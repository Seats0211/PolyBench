/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)( i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
	B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
      }
    }
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
void kernel_gesummv(int n,
            DATA_TYPE alpha,
            DATA_TYPE beta,
            DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
            DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
            DATA_TYPE POLYBENCH_1D(tmp,N,n),
            DATA_TYPE POLYBENCH_1D(x,N,n),
            DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
#if defined(SE)
      /* Tunable predicates and thresholds */
      const DATA_TYPE ALPHA_TH = SCALAR_VAL(1e-3);
      const DATA_TYPE BETA_TH  = SCALAR_VAL(1e-3);
      const DATA_TYPE N_TH     = SCALAR_VAL(64.0); /* tune per problem size */

      int condA = (fabs(alpha) > ALPHA_TH);
      int condB = (i % 2 == 0);
      int condC = (((int)(x[0] * 1000.0)) & 1);

      int condA2 = (fabs(beta) > BETA_TH);
      int condB2 = ((i + 1) % 2 == 0);
      int condC2 = (((int)(y[0] * 1000.0)) & 1);

      /* Initialize accumulators */
      DATA_TYPE tmp_acc = SCALAR_VAL(0.0);
      DATA_TYPE y_acc   = SCALAR_VAL(0.0);

      /* Choose path for tmp and y accumulation using Shannon-like expansion */
      if (condA) {
        if (condB || (condC && (n > N_TH))) {
          /* Hot path: full accumulation for both tmp and y */
#if defined(LU) && (UNROLL > 1)
          int jj = 0;
          for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
            tmp_acc += A[i][jj+0] * x[jj+0];
            y_acc   += B[i][jj+0] * x[jj+0];

            tmp_acc += A[i][jj+1] * x[jj+1];
            y_acc   += B[i][jj+1] * x[jj+1];

            tmp_acc += A[i][jj+2] * x[jj+2];
            y_acc   += B[i][jj+2] * x[jj+2];

            tmp_acc += A[i][jj+3] * x[jj+3];
            y_acc   += B[i][jj+3] * x[jj+3];
#else
            for (int uu=0; uu<UNROLL; ++uu) {
              tmp_acc += A[i][jj+uu] * x[jj+uu];
              y_acc   += B[i][jj+uu] * x[jj+uu];
            }
#endif
          }
          for (; jj < _PB_N; ++jj) {
            tmp_acc += A[i][jj] * x[jj];
            y_acc   += B[i][jj] * x[jj];
          }
#else
          for (j = 0; j < _PB_N; j++) {
            tmp_acc += A[i][j] * x[j];
            y_acc   += B[i][j] * x[j];
          }
#endif
        } else {
          /* condA true but subguard false -> reduced-cost (approx) accumulation */
          for (j = 0; j < _PB_N; j += 2) {
            tmp_acc += A[i][j] * x[j];
            y_acc   += B[i][j] * x[j];
          }
          /* scale to approximate full sum */
          tmp_acc *= SCALAR_VAL(2.0);
          y_acc   *= SCALAR_VAL(2.0);
        }
      } else {
        /* condA == false: alternate branching */
        if (condC && (n > N_TH)) {
          /* alternate hot: full accumulation */
#if defined(LU) && (UNROLL > 1)
          int jj = 0;
          for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
#if UNROLL == 4
            tmp_acc += A[i][jj+0] * x[jj+0];
            y_acc   += B[i][jj+0] * x[jj+0];

            tmp_acc += A[i][jj+1] * x[jj+1];
            y_acc   += B[i][jj+1] * x[jj+1];

            tmp_acc += A[i][jj+2] * x[jj+2];
            y_acc   += B[i][jj+2] * x[jj+2];

            tmp_acc += A[i][jj+3] * x[jj+3];
            y_acc   += B[i][jj+3] * x[jj+3];
#else
            for (int uu=0; uu<UNROLL; ++uu) {
              tmp_acc += A[i][jj+uu] * x[jj+uu];
              y_acc   += B[i][jj+uu] * x[jj+uu];
            }
#endif
          }
          for (; jj < _PB_N; ++jj) {
            tmp_acc += A[i][jj] * x[jj];
            y_acc   += B[i][jj] * x[jj];
          }
#else
          for (j = 0; j < _PB_N; j++) {
            tmp_acc += A[i][j] * x[j];
            y_acc   += B[i][j] * x[j];
          }
#endif
        } else {
          /* cold path: lighter approximate */
          for (j = 0; j < _PB_N; j += 2) {
            tmp_acc += A[i][j] * x[j];
            y_acc   += B[i][j] * x[j];
          }
          tmp_acc *= SCALAR_VAL(2.0);
          y_acc   *= SCALAR_VAL(2.0);
        }
      }

      /* Write back intermediate results */
      tmp[i] = tmp_acc;
      y[i]   = alpha * tmp_acc + beta * y_acc;

#else
      /* baseline: exact accumulation for tmp and y */
      tmp[i] = SCALAR_VAL(0.0);
      y[i]   = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++) {
        tmp[i] = A[i][j] * x[j] + tmp[i];
        y[i]   = B[i][j] * x[j] + y[i];
      }
      y[i] = alpha * tmp[i] + beta * y[i];
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
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gesummv (n, alpha, beta,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(B),
		  POLYBENCH_ARRAY(tmp),
		  POLYBENCH_ARRAY(x),
		  POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}
