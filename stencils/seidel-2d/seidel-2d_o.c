/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* seidel-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "seidel-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
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
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_seidel_2d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;

#pragma scop
  
  const int SAMPLE_LEN_TH = 512;
  const int SAMPLE_STEP = 2;

  for (t = 0; t <= _PB_TSTEPS - 1; t++) {
#if defined(SE)
    int use_sampling = (_PB_N > SAMPLE_LEN_TH);

    if (!use_sampling) {
      
  #if defined(LU) && (UNROLL > 1)
      
      int ii = 1;
      for (; ii + (UNROLL - 1) < _PB_N - 1; ii += UNROLL) {
        for (j = 1; j <= _PB_N - 2; ++j) {
  #if UNROLL == 4
          A[ii  ][j] = (A[ii-1][j-1] + A[ii-1][j] + A[ii-1][j+1]
                       + A[ii  ][j-1] + A[ii  ][j] + A[ii  ][j+1]
                       + A[ii+1][j-1] + A[ii+1][j] + A[ii+1][j+1]) / SCALAR_VAL(9.0);
          A[ii+1][j] = (A[ii  ][j-1] + A[ii  ][j] + A[ii  ][j+1]
                       + A[ii+1][j-1] + A[ii+1][j] + A[ii+1][j+1]
                       + A[ii+2][j-1] + A[ii+2][j] + A[ii+2][j+1]) / SCALAR_VAL(9.0);
          A[ii+2][j] = (A[ii+1][j-1] + A[ii+1][j] + A[ii+1][j+1]
                       + A[ii+2][j-1] + A[ii+2][j] + A[ii+2][j+1]
                       + A[ii+3][j-1] + A[ii+3][j] + A[ii+3][j+1]) / SCALAR_VAL(9.0);
          A[ii+3][j] = (A[ii+2][j-1] + A[ii+2][j] + A[ii+2][j+1]
                       + A[ii+3][j-1] + A[ii+3][j] + A[ii+3][j+1]
                       + A[ii+4][j-1] + A[ii+4][j] + A[ii+4][j+1]) / SCALAR_VAL(9.0);
  #else
          for (int uu = 0; uu < UNROLL; ++uu) {
            int ii_local = ii + uu;
            A[ii_local][j] = (A[ii_local-1][j-1] + A[ii_local-1][j] + A[ii_local-1][j+1]
                              + A[ii_local  ][j-1] + A[ii_local  ][j] + A[ii_local  ][j+1]
                              + A[ii_local+1][j-1] + A[ii_local+1][j] + A[ii_local+1][j+1]) / SCALAR_VAL(9.0);
          }
  #endif
        }
      }
      for (; ii < _PB_N - 1; ++ii)
        for (j = 1; j <= _PB_N - 2; ++j)
          A[ii][j] = (A[ii-1][j-1] + A[ii-1][j] + A[ii-1][j+1]
                   + A[ii  ][j-1] + A[ii  ][j] + A[ii  ][j+1]
                   + A[ii+1][j-1] + A[ii+1][j] + A[ii+1][j+1]) / SCALAR_VAL(9.0);
  #else
      for (i = 1; i <= _PB_N - 2; i++)
        for (j = 1; j <= _PB_N - 2; j++)
          A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                   + A[i  ][j-1] + A[i  ][j] + A[i  ][j+1]
                   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / SCALAR_VAL(9.0);
  #endif
    } else {
      
      for (i = 1; i <= _PB_N - 2; ++i) {
        for (j = 1; j <= _PB_N - 2; ++j) {
          int condHot = (((i + j) % SAMPLE_STEP) == 0);
          if (condHot) {
            A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                     + A[i  ][j-1] + A[i  ][j] + A[i  ][j+1]
                     + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / SCALAR_VAL(9.0);
          } else {
            
            A[i][j] = SCALAR_VAL(0.0);
          }
        }
      }

      
      for (i = 1; i <= _PB_N - 2; ++i) {
        for (j = 1; j <= _PB_N - 2; ++j) {
          if (A[i][j] == SCALAR_VAL(0.0)) {
            DATA_TYPE s = SCALAR_VAL(0.0);
            int cnt = 0;
            
            int ii, jj;
            for (ii = i-1; ii <= i+1; ++ii) {
              for (jj = j-1; jj <= j+1; ++jj) {
                if (ii==i && jj==j) continue;
                if (((ii + jj) % SAMPLE_STEP) == 0) {
                  s += A[ii][jj];
                  cnt++;
                }
              }
            }
            if (cnt > 0) {
              A[i][j] = s / (DATA_TYPE)cnt;
            } else {
              
              A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                       + A[i  ][j-1] + A[i  ][j] + A[i  ][j+1]
                       + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / SCALAR_VAL(9.0);
            }
          }
        }
      }
    } /* end use_sampling */
#else
    /* baseline exact (no SE) */
    for (i = 1; i <= _PB_N - 2; i++)
      for (j = 1; j <= _PB_N - 2; j++)
        A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                 + A[i  ][j-1] + A[i  ][j] + A[i  ][j+1]
                 + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / SCALAR_VAL(9.0);
#endif

  } /* end timesteps */
#pragma endscop

}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_seidel_2d (tsteps, n, POLYBENCH_ARRAY(A));

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

