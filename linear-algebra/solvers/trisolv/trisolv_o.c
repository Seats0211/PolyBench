/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trisolv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = - 999;
      b[i] =  i ;
      for (j = 0; j <= i; j++)
	L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n;
    }
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
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trisolv(int n,
            DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
            DATA_TYPE POLYBENCH_1D(x,N,n),
            DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      /* initialize with RHS */
      x[i] = b[i];

#if defined(SE)
      /* Shannon-style composite guard parameters (tunable) */
      const DATA_TYPE X_TH = SCALAR_VAL(1e-6); /* cheap threshold on x or L */
      const int I_TH = 64;                    /* problem-size threshold */
      int condStruct = (i % 2 == 0);          /* structural predicate (cheap) */
      int condRand;                           /* pseudo-randomish for variety */

      /* Choose predicate based on current row to decide hot/cold path */
      condRand = (((int)(L[i][0] * 1000.0)) & 1);

      DATA_TYPE acc = SCALAR_VAL(0.0);

      if (condStruct || (condRand && (_PB_N > I_TH))) {
        /* hot path: full accurate accumulation (optionally unrolled) */
  #if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < i; jj += UNROLL) {
  #if UNROLL == 4
          acc += L[i][jj+0] * x[jj+0];
          acc += L[i][jj+1] * x[jj+1];
          acc += L[i][jj+2] * x[jj+2];
          acc += L[i][jj+3] * x[jj+3];
  #else
          for (int uu=0; uu<UNROLL; ++uu)
            acc += L[i][jj+uu] * x[jj+uu];
  #endif
        }
        for (; jj < i; ++jj) acc += L[i][jj] * x[jj];
  #else
        for (j = 0; j < i; j++)
          acc += L[i][j] * x[j];
  #endif
      } else {
        /* cold path: sample every 2 (very coarse) and compensate */
  #if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < i; jj += UNROLL) {
  #if UNROLL == 4
          /* when unrolling, still step by UNROLL but we will sample only even offsets inside */
          acc += L[i][jj+0] * x[jj+0];
          /* skip jj+1 */
          acc += L[i][jj+2] * x[jj+2];
          /* skip jj+3 */
  #else
          for (int uu=0; uu<UNROLL; ++uu) {
            if (((jj+uu) % 2) == 0) acc += L[i][jj+uu] * x[jj+uu];
          }
  #endif
        }
        for (; jj < i; ++jj) if ((jj % 2) == 0) acc += L[i][jj] * x[jj];
  #else
        for (j = 0; j < i; j += 2)
          acc += L[i][j] * x[j];
  #endif
        acc *= SCALAR_VAL(2.0); /* crude compensation */
      }

      /* subtract accumulated value from RHS */
      x[i] -= acc;

#else
      /* baseline exact accumulation */
      for (j = 0; j < i; j++)
        x[i] -= L[i][j] * x[j];
#endif

      /* exact division by diagonal element (must be precise) */
      if (fabsl(L[i][i]) < SCALAR_VAL(1e-18)) {
        /* clamp tiny pivot to avoid blow-up; record in experiments if occurs */
        L[i][i] = (L[i][i] >= 0) ? SCALAR_VAL(1e-18) : -SCALAR_VAL(1e-18);
      }
      x[i] = x[i] / L[i][i];
    }
#pragma endscop

}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trisolv (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);

  return 0;
}
