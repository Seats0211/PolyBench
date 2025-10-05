/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_bicg(int m, int n,
                 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                 DATA_TYPE POLYBENCH_1D(s,M,m),
                 DATA_TYPE POLYBENCH_1D(q,N,n),
                 DATA_TYPE POLYBENCH_1D(p,M,m),
                 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

#pragma scop
  /* initialize s */
  for (j = 0; j < _PB_M; j++)
    s[j] = SCALAR_VAL(0.0);

  /* main loops */
  for (i = 0; i < _PB_N; i++)
  {
#if defined(SE)
    /* Shannon-style cheap predicates */
    const DATA_TYPE R_TH = SCALAR_VAL(1e-3);
    const int M_TH = 64;  /* problem-size threshold for hot/cold path */

    int cond_r = (fabs(r[i]) > R_TH);
    int cond_i = (i % 2 == 0);
    int cond_a = (((int)(A[i][0]*1000.0)) & 1); /* pseudo-random */

    DATA_TYPE q_acc = SCALAR_VAL(0.0);

    for (j = 0; j < _PB_M; )
    {
      if (cond_r) {
        if (cond_i || (cond_a && (_PB_M > M_TH))) {
          /* hot path: full accumulation */
#if defined(LU) && (UNROLL > 1)
          int jj = j;
          for (; jj + (UNROLL-1) < _PB_M; jj += UNROLL) {
#if UNROLL == 4
            s[jj+0] += r[i] * A[i][jj+0];
            s[jj+1] += r[i] * A[i][jj+1];
            s[jj+2] += r[i] * A[i][jj+2];
            s[jj+3] += r[i] * A[i][jj+3];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              s[jj+uu] += r[i]*A[i][jj+uu];
#endif
          }
          for (; jj < _PB_M; ++jj)
            s[jj] += r[i]*A[i][jj];
          j = jj;
#else
          for (; j < _PB_M; ++j)
            s[j] += r[i]*A[i][j];
#endif
        } else {
          /* cold path: sample every 2 */
          for (; j < _PB_M; j += 2)
            s[j] += r[i]*A[i][j];
          /* rough compensation */
          for (int jj=0; jj < _PB_M; jj += 2)
            s[jj] *= SCALAR_VAL(2.0);
        }
      } else {
        if (cond_a && (_PB_M > M_TH)) {
#if defined(LU) && (UNROLL > 1)
          int jj = j;
          for (; jj + (UNROLL-1) < _PB_M; jj += UNROLL) {
#if UNROLL == 4
            s[jj+0] += r[i] * A[i][jj+0];
            s[jj+1] += r[i] * A[i][jj+1];
            s[jj+2] += r[i] * A[i][jj+2];
            s[jj+3] += r[i] * A[i][jj+3];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              s[jj+uu] += r[i]*A[i][jj+uu];
#endif
          }
          for (; jj < _PB_M; ++jj)
            s[jj] += r[i]*A[i][jj];
          j = jj;
#else
          for (; j < _PB_M; ++j)
            s[j] += r[i]*A[i][j];
#endif
        } else {
          /* cold path: sample every 2 */
          for (; j < _PB_M; j += 2)
            s[j] += r[i]*A[i][j];
          for (int jj=0; jj < _PB_M; jj += 2)
            s[jj] *= SCALAR_VAL(2.0);
        }
      }
    }

    /* compute q[i] */
    DATA_TYPE q_tmp = SCALAR_VAL(0.0);
    if (cond_r) {
      if (cond_i || (cond_a && (_PB_M > M_TH))) {
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL-1) < _PB_M; jj += UNROLL) {
#if UNROLL == 4
          q_tmp += A[i][jj+0] * p[jj+0];
          q_tmp += A[i][jj+1] * p[jj+1];
          q_tmp += A[i][jj+2] * p[jj+2];
          q_tmp += A[i][jj+3] * p[jj+3];
#else
          for (int uu=0; uu<UNROLL; ++uu)
            q_tmp += A[i][jj+uu]*p[jj+uu];
#endif
        }
        for (; jj < _PB_M; ++jj)
          q_tmp += A[i][jj]*p[jj];
#else
        for (j=0; j<_PB_M; ++j)
          q_tmp += A[i][j]*p[j];
#endif
      } else {
        for (j=0; j<_PB_M; j+=2)
          q_tmp += A[i][j]*p[j];
        q_tmp *= SCALAR_VAL(2.0);
      }
    } else {
      if (cond_a && (_PB_M > M_TH)) {
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL-1) < _PB_M; jj += UNROLL) {
#if UNROLL == 4
          q_tmp += A[i][jj+0] * p[jj+0];
          q_tmp += A[i][jj+1] * p[jj+1];
          q_tmp += A[i][jj+2] * p[jj+2];
          q_tmp += A[i][jj+3] * p[jj+3];
#else
          for (int uu=0; uu<UNROLL; ++uu)
            q_tmp += A[i][jj+uu]*p[jj+uu];
#endif
        }
        for (; jj < _PB_M; ++jj)
          q_tmp += A[i][jj]*p[jj];
#else
        for (j=0; j<_PB_M; ++j)
          q_tmp += A[i][j]*p[j];
#endif
      } else {
        for (j=0; j<_PB_M; j+=2)
          q_tmp += A[i][j]*p[j];
        q_tmp *= SCALAR_VAL(2.0);
      }
    }
    q[i] = q_tmp;

#else
    /* baseline: exact compute */
    q[i] = SCALAR_VAL(0.0);
    for (j=0; j<_PB_M; j++)
    {
      s[j] += r[i]*A[i][j];
      q[i] += A[i][j]*p[j];
    }
#endif
  }
#pragma endscop
}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_bicg (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(s),
	       POLYBENCH_ARRAY(q),
	       POLYBENCH_ARRAY(p),
	       POLYBENCH_ARRAY(r));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}
