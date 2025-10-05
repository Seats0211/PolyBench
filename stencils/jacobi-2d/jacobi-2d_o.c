/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"


/* Array initialization. */
static
void init_array (int n,
         DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
         DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
    A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
    B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
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


/* Main computational kernel with Strategy B */
static
void kernel_jacobi_2d(int tsteps,
                int n,
                DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int t, i, j;

#pragma scop
  /* Tuning params for Strategy B */
  const int SAMPLE_LEN_TH = 512; /* enable sampling for large problems */
  const int SAMPLE_STEP = 2;     /* checkerboard sampling stride for cold paths */
  const int UNROLL = 4;          /* unroll factor for hot paths */

  for (t = 0; t < _PB_TSTEPS; t++)
    {
      /* Update B from A: Strategy B application */
#if defined(SE)
      int use_sampling = (_PB_N > SAMPLE_LEN_TH);

      if (!use_sampling) {
        /* Exact baseline with optional unroll on hot paths */
#if defined(LU)
        int ii = 1;
        for (; ii + (UNROLL - 1) < _PB_N - 1; ii += UNROLL) {
          for (j = 1; j < _PB_N - 1; ++j) {
#if UNROLL == 4
            /* Hot path: exact calculation with unroll */
            B[ii  ][j] = SCALAR_VAL(0.2) * (A[ii  ][j] + A[ii  ][j-1] + A[ii  ][j+1] + A[ii-1][j] + A[ii+1][j]);
            B[ii+1][j] = SCALAR_VAL(0.2) * (A[ii+1][j] + A[ii+1][j-1] + A[ii+1][j+1] + A[ii  ][j] + A[ii+2][j]);
            B[ii+2][j] = SCALAR_VAL(0.2) * (A[ii+2][j] + A[ii+2][j-1] + A[ii+2][j+1] + A[ii+1][j] + A[ii+3][j]);
            B[ii+3][j] = SCALAR_VAL(0.2) * (A[ii+3][j] + A[ii+3][j-1] + A[ii+3][j+1] + A[ii+2][j] + A[ii+4][j]);
#else
            for (int uu = 0; uu < UNROLL; ++uu) {
              int ii_local = ii + uu;
              B[ii_local][j] = SCALAR_VAL(0.2) * (A[ii_local][j] + A[ii_local][j-1] + A[ii_local][j+1]
                                                 + A[ii_local-1][j] + A[ii_local+1][j]);
            }
#endif
          }
        }
        for (; ii < _PB_N - 1; ++ii)
          for (j = 1; j < _PB_N - 1; ++j)
            B[ii][j] = SCALAR_VAL(0.2) * (A[ii][j] + A[ii][j-1] + A[ii][j+1] + A[ii-1][j] + A[ii+1][j]);
#else
        /* Baseline exact */
        for (i = 1; i < _PB_N - 1; i++)
          for (j = 1; j < _PB_N - 1; j++)
            B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i-1][j] + A[i+1][j]);
#endif
      } else {
        /* Large problem: Shannon-style branch split for hot/cold paths */
        for (i = 1; i < _PB_N - 1; ++i) {
          for (j = 1; j < _PB_N - 1; ++j) {
            /* Shannon-style split: Pivot on is_hot (composite condition) */
            DATA_TYPE center_val = A[i][j];
            DATA_TYPE diff_h = fabs(A[i][j] - A[i][j+1]);  /* Horizontal diff */
            DATA_TYPE diff_v = fabs(A[i][j] - A[i+1][j]);  /* Vertical diff */
            DATA_TYPE threshold_h = SCALAR_VAL(0.1);
            DATA_TYPE threshold_v = SCALAR_VAL(0.1);
            DATA_TYPE threshold_center = SCALAR_VAL(0.5);
            
            /* Pivot A: center_val > threshold_center (high-frequency hot path ~60%) */
            int pivot_A = (center_val > threshold_center);
            if (pivot_A) {
              /* Hot path: Exact calculation */
              B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i-1][j] + A[i+1][j]);
            } else {
              /* Cold path: Further split on diffs (Shannon decomposition) */
              int B_part = (diff_h < threshold_h || diff_v < threshold_v);  /* Simple OR for cold refinement */
              if (B_part) {
                /* Cold sub-path 1: Approximate with average of neighbors (sampling) */
                DATA_TYPE sum_neighbors = A[i][j-1] + A[i][j+1] + A[i-1][j] + A[i+1][j];
                B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + sum_neighbors * SCALAR_VAL(0.25));
              } else {
                /* Cold sub-path 2: Checkerboard sampling approximation */
                int is_sample = (((i + j) % SAMPLE_STEP) == 0);
                if (is_sample) {
                  /* Sample exact */
                  B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i-1][j] + A[i+1][j]);
                } else {
                  /* Interpolate from sampled neighbors */
                  DATA_TYPE s = SCALAR_VAL(0.0);
                  int cnt = 0;
                  for (int ii = i-1; ii <= i+1; ++ii)
                    for (int jj = j-1; jj <= j+1; ++jj) {
                      if (ii == i && jj == j) continue;
                      if (((ii + jj) % SAMPLE_STEP) == 0) {
                        s += A[ii][jj];
                        cnt++;
                      }
                    }
                  if (cnt > 0) B[i][j] = s / (DATA_TYPE)cnt * SCALAR_VAL(0.2);
                  else B[i][j] = A[i][j];  /* Fallback */
                }
              }
            }
          }
        }
      }
#else
      /* Baseline exact (unchanged) */
      for (i = 1; i < _PB_N - 1; i++)
    for (j = 1; j < _PB_N - 1; j++)
      B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i-1][j] + A[i+1][j]);
#endif

      /* Swap A and B: Optional unroll (unchanged from original) */
#if defined(LU) && (UNROLL > 1)
      {
        int ii = 1;
        for (; ii + (UNROLL - 1) < _PB_N - 1; ii += UNROLL) {
          for (j = 1; j < _PB_N - 1; ++j) {
#if UNROLL == 4
            A[ii  ][j] = B[ii  ][j];
            A[ii+1][j] = B[ii+1][j];
            A[ii+2][j] = B[ii+2][j];
            A[ii+3][j] = B[ii+3][j];
#else
            for (int uu = 0; uu < UNROLL; ++uu) {
              A[ii + uu][j] = B[ii + uu][j];
            }
#endif
          }
        }
        for (; ii < _PB_N - 1; ++ii)
          for (j = 1; j < _PB_N - 1; ++j)
            A[ii][j] = B[ii][j];
      }
#else
      for (i = 1; i < _PB_N - 1; i++)
    for (j = 1; j < _PB_N - 1; j++)
      A[i][j] = B[i][j];
#endif
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}