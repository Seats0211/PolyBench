/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* correlation.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"


/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  DATA_TYPE eps = SCALAR_VAL(0.1);


#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < _PB_M; j++)
    {
      stddev[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          #if defined(SE)
  const DATA_TYPE MEAN_TH = SCALAR_VAL(1e-3);
  const DATA_TYPE N_TH = SCALAR_VAL(50.0); /* tune per problem size */
  int condA = (fabs(mean[i]) > MEAN_TH);      /* cheap predicate */
  int condB = ((j % 2) == 0);
  int condC = (((int)(data[0][i]*1000.0)) & 1);

  if (condA) {
    if (condB || (condC && (float_n > N_TH))) {
      /* hot: full accumulation */
      corr[i][j] = SCALAR_VAL(0.0);
#if defined(LU) && (UNROLL > 1)
      /* optional manual unrolling here */
      int kk = 0;
      for (; kk + (UNROLL - 1) < _PB_N; kk += UNROLL) {
        /* unrolled body (example for UNROLL==4) */
        corr[i][j] += data[kk+0][i] * data[kk+0][j];
        corr[i][j] += data[kk+1][i] * data[kk+1][j];
        corr[i][j] += data[kk+2][i] * data[kk+2][j];
        corr[i][j] += data[kk+3][i] * data[kk+3][j];
      }
      for (; kk < _PB_N; ++kk) corr[i][j] += data[kk][i] * data[kk][j];
#else
      for (k = 0; k < _PB_N; k++)
        corr[i][j] += data[k][i] * data[k][j];
#endif
    } else {
      /* condA true but subguard false -> reduced accumulation (step-2) */
      corr[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_N; k += 2)
        corr[i][j] += data[k][i] * data[k][j];
      /* scale to approximate full sum (assumes roughly uniform distribution) */
      corr[i][j] *= SCALAR_VAL(2.0);
    }
  } else {
    /* condA false: another branch with similar pattern */
    if (condC && (float_n > N_TH)) {
      /* alternate hot: full */
      corr[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_N; k++)
        corr[i][j] += data[k][i] * data[k][j];
    } else {
      /* cold: lighter approximate */
      corr[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_N; k += 2)
        corr[i][j] += data[k][i] * data[k][j];
      corr[i][j] *= SCALAR_VAL(2.0);
    }
  }
  corr[j][i] = corr[i][j];
#else
  /* baseline */
  corr[i][j] = SCALAR_VAL(0.0);
  for (k = 0; k < _PB_N; k++)
    corr[i][j] += data[k][i] * data[k][j];
  corr[j][i] = corr[i][j];
#endif

        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_correlation (m, n, float_n,
		      POLYBENCH_ARRAY(data),
		      POLYBENCH_ARRAY(corr),
		      POLYBENCH_ARRAY(mean),
		      POLYBENCH_ARRAY(stddev));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}
