/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"


#define SCALAR_VAL(x) ((DATA_TYPE)(x))


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_floyd_warshall(int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

#pragma scop
  /* Tuning parameters (can be adjusted for experiments) */
  const DATA_TYPE INF_TH = SCALAR_VAL(1e3); /* values >= this treated as 'infinite' */
  const int SAMPLE_LEN_TH = 512;           /* enable sampling for large problems */
  const int SAMPLE_STEP = 2;               /* sampling stride (checkerboard) */

  for (k = 0; k < _PB_N; k++)
    {
      for (i = 0; i < _PB_N; i++)
	{
#if defined(SE)
	  int use_sampling = (_PB_N > SAMPLE_LEN_TH);

	  if (!use_sampling) {
	    /* Small problem or sampling disabled: do the exact baseline update.
	       Optional manual unroll for inner j loop. */
#if defined(LU) && (UNROLL > 1)
	    int jj = 0;
	    for (; jj + (UNROLL - 1) < _PB_N; jj += UNROLL) {
	      /* unrolled block */
	      for (int uu = 0; uu < UNROLL; ++uu) {
		int jj_idx = jj + uu;
		DATA_TYPE tmp = path[i][k] + path[k][jj_idx];
		/* keep the minimum */
		path[i][jj_idx] = (path[i][jj_idx] < tmp) ? path[i][jj_idx] : tmp;
	      }
	    }
	    for (; jj < _PB_N; ++jj) {
	      DATA_TYPE tmp = path[i][k] + path[k][jj];
	      path[i][jj] = (path[i][jj] < tmp) ? path[i][jj] : tmp;
	    }
#else
	    for (j = 0; j < _PB_N; j++) {
	      DATA_TYPE tmp = path[i][k] + path[k][j];
	      path[i][j] = (path[i][j] < tmp) ? path[i][j] : tmp;
	    }
#endif
	  } else {
	    /* Large problem: hot/cold split with sampling on cold positions.
	       hot: tmp < INF_TH -> exact update
	       cold: only update sampled positions (i+j)%SAMPLE_STEP==0 to reduce work */
	    for (j = 0; j < _PB_N; j++) {
	      DATA_TYPE tmp = path[i][k] + path[k][j];
	      int hot = (tmp < INF_TH);
	      if (hot) {
		/* exact */
		path[i][j] = (path[i][j] < tmp) ? path[i][j] : tmp;
	      } else {
		/* cold: sample update on checkerboard positions */
		if ( ((i + j) % SAMPLE_STEP) == 0 ) {
		  path[i][j] = (path[i][j] < tmp) ? path[i][j] : tmp;
		} else {
		  /* leave path[i][j] unchanged (approximation) */
		}
	      }
	    }
	  }
#else
	  /* baseline exact (SE not defined) */
	  for (j = 0; j < _PB_N; j++) {
	    DATA_TYPE tmp = path[i][k] + path[k][j];
	    path[i][j] = (path[i][j] < tmp) ? path[i][j] : tmp;
	  }
#endif /* SE */
	}
    }
#pragma endscop

}





int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_floyd_warshall (n, POLYBENCH_ARRAY(path));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);

  return 0;
}
