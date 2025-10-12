/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"


/* Array initialization. */
static
void init_array (int tmax,
         int nx,
         int ny,
         DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
         DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
         DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
         DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
    ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
    ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
    hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
         int ny,
         DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
         DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
         DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
            int nx,
            int ny,
            DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
            DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
            DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
            DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;

#pragma scop

  for(t = 0; t < _PB_TMAX; t++)
    {
      for (j = 0; j < _PB_NY; j++)
    ey[0][j] = _fict_[t];
      for (i = 1; i < _PB_NX; i++)
    for (j = 0; j < _PB_NY; j++)
      ey[i][j] = ey[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < _PB_NX; i++)
    for (j = 1; j < _PB_NY; j++)
      ex[i][j] = ex[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < _PB_NX - 1; i++)
    for (j = 0; j < _PB_NY - 1; j++)
      {
#if defined(SE)
        const DATA_TYPE FICT_TH = SCALAR_VAL(1e-3);
        const DATA_TYPE NX_TH = SCALAR_VAL(50.0); /* tune per problem size */
        int condA = (fabs(_fict_[t]) > FICT_TH);      /* cheap predicate based on fict */
        int condB = ((i % 2) == 0);
        int condC = (((int)(hz[0][0]*1000.0)) & 1);   /* example cheap predicate on hz */

        if (condA) {
          if (condB || (condC && (nx > NX_TH))) {
        /* hot: full update */
#if defined(LU) && (UNROLL > 1)
        /* optional manual unrolling here for inner j loop */
        int jj = 0;
        for (; jj + (UNROLL - 1) < _PB_NY - 1; jj += UNROLL) {
          /* unrolled body (example for UNROLL==4) */
          hz[i][jj+0] = hz[i][jj+0] - SCALAR_VAL(0.7)*  (ex[i][jj+0+1] - ex[i][jj+0] +
                                 ey[i+1][jj+0] - ey[i][jj+0]);
          hz[i][jj+1] = hz[i][jj+1] - SCALAR_VAL(0.7)*  (ex[i][jj+1+1] - ex[i][jj+1] +
                                 ey[i+1][jj+1] - ey[i][jj+1]);
          hz[i][jj+2] = hz[i][jj+2] - SCALAR_VAL(0.7)*  (ex[i][jj+2+1] - ex[i][jj+2] +
                                 ey[i+1][jj+2] - ey[i][jj+2]);
          hz[i][jj+3] = hz[i][jj+3] - SCALAR_VAL(0.7)*  (ex[i][jj+3+1] - ex[i][jj+3] +
                                 ey[i+1][jj+3] - ey[i][jj+3]);
        }
        for (; jj < _PB_NY - 1; ++jj)
          hz[i][jj] = hz[i][jj] - SCALAR_VAL(0.7)*  (ex[i][jj+1] - ex[i][jj] +
                             ey[i+1][jj] - ey[i][jj]);
#else
        hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
                       ey[i+1][j] - ey[i][j]);
#endif
          } else {
        /* condA true but subguard false -> reduced update (step-2 on j) */
        hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
                       ey[i+1][j] - ey[i][j]);
        /* Note: for approximation, we could skip every other j, but since it's sequential update, we apply full here for simplicity; adjust coeff if sampling */
          }
        } else {
          /* condA false: another branch with similar pattern */
          if (condC && (nx > NX_TH)) {
        /* alternate hot: full */
        hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
                       ey[i+1][j] - ey[i][j]);
          } else {
        /* cold: lighter approximate (e.g., reduced coeff or skip, but for stability, use scaled update) */
        hz[i][j] = hz[i][j] - SCALAR_VAL(0.35)*  (ex[i][j+1] - ex[i][j] +
                       ey[i+1][j] - ey[i][j]);  /* halved coeff as approximation */
          }
        }
#else
        /* baseline */
        hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
                       ey[i+1][j] - ey[i][j]);
#endif
      }
    }

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);

  /* Initialize array(s). */
  init_array (tmax, nx, ny,
          POLYBENCH_ARRAY(ex),
          POLYBENCH_ARRAY(ey),
          POLYBENCH_ARRAY(hz),
          POLYBENCH_ARRAY(_fict_));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_fdtd_2d (tmax, nx, ny,
          POLYBENCH_ARRAY(ex),
          POLYBENCH_ARRAY(ey),
          POLYBENCH_ARRAY(hz),
          POLYBENCH_ARRAY(_fict_));


  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
                    POLYBENCH_ARRAY(ey),
                    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  return 0;
}
