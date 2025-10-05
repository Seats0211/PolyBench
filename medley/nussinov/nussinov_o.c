/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* nussinov.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "nussinov.h"

/* RNA bases represented as chars, range is [0,3] */
typedef char base;

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

/* Array initialization. */
static
void init_array (int n,
                 base POLYBENCH_1D(seq,N,n),
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j;

  //base is AGCT/0..3
  for (i=0; i <n; i++) {
     seq[i] = (base)((i+1)%4);
  }

  for (i=0; i <n; i++)
     for (j=0; j <n; j++)
       table[i][j] = 0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))

{
  int i, j;
  int t = 0;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("table");
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, table[i][j]);
      t++;
    }
  }
  POLYBENCH_DUMP_END("table");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/*
  Original version by Dave Wonnacott at Haverford College <davew@cs.haverford.edu>,
  with help from Allison Lake, Ting Zhou, and Tian Jin,
  based on algorithm by Nussinov, described in Allison Lake's senior thesis.
*/
static
void kernel_nussinov(int n, base POLYBENCH_1D(seq,N,n),
                     DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j, k;

#pragma scop
  /* Strategy B:
     - Use Shannon-style composite guard around the expensive inner split loop.
     - Hot path: full scan of k (exact).
     - Cold path: sampled scan of k (e.g., step=2) as coarse approximation.
     - Tunables: LEN_TH (when interval length large), UNROLL for manual unroll.
  */
  const int LEN_TH = 64; /* threshold on interval length to consider sampling */
  for (i = _PB_N - 1; i >= 0; i--) {
    for (j = i + 1; j < _PB_N; j++) {

      /* keep small/local updates exact */
      if (j - 1 >= 0)
        table[i][j] = max_score(table[i][j], table[i][j-1]);
      if (i + 1 < _PB_N)
        table[i][j] = max_score(table[i][j], table[i+1][j]);

      if (j - 1 >= 0 && i + 1 < _PB_N) {
        if (i < j - 1)
          table[i][j] = max_score(table[i][j], table[i+1][j-1] + match(seq[i], seq[j]));
        else
          table[i][j] = max_score(table[i][j], table[i+1][j-1]);
      }

      /* expensive split loop: apply SE-style decision */
#if defined(SE)
      {
        int len = j - i;
        /* condA: long interval (candidate for sampling) */
        int condA = (len > LEN_TH);
        /* condB: structural inexpensive predicate (parity) */
        int condB = (((i + j) & 1) == 0);
        /* condC: pseudo-random bit to diversify behavior */
        int condC = (((int)(seq[i] + seq[j]) & 1));

        if (!condA) {
          /* short intervals: do full exact scan (hot) */
#if defined(LU) && (UNROLL > 1)
          int kk = i + 1;
          for (; kk + (UNROLL - 1) < j; kk += UNROLL) {
#if UNROLL == 4
            table[i][j] = max_score(table[i][j], table[i][kk] + table[kk+1][j]);
            table[i][j] = max_score(table[i][j], table[i][kk+1] + table[kk+2][j]);
            table[i][j] = max_score(table[i][j], table[i][kk+2] + table[kk+3][j]);
            table[i][j] = max_score(table[i][j], table[i][kk+3] + table[kk+4][j]);
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              table[i][j] = max_score(table[i][j], table[i][kk+uu] + table[kk+uu+1][j]);
#endif
          }
          for (; kk < j; ++kk)
            table[i][j] = max_score(table[i][j], table[i][kk] + table[kk+1][j]);
#else
          for (k = i + 1; k < j; k++)
            table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
#endif
        } else {
          /* long intervals: apply Shannon expansion on condB */
          if (condB || (condC && (_PB_N > LEN_TH))) {
            /* likely-hot subcase: do full scan */
#if defined(LU) && (UNROLL > 1)
            int kk = i + 1;
            for (; kk + (UNROLL - 1) < j; kk += UNROLL) {
#if UNROLL == 4
              table[i][j] = max_score(table[i][j], table[i][kk] + table[kk+1][j]);
              table[i][j] = max_score(table[i][j], table[i][kk+1] + table[kk+2][j]);
              table[i][j] = max_score(table[i][j], table[i][kk+2] + table[kk+3][j]);
              table[i][j] = max_score(table[i][j], table[i][kk+3] + table[kk+4][j]);
#else
              for (int uu = 0; uu < UNROLL; ++uu)
                table[i][j] = max_score(table[i][j], table[i][kk+uu] + table[kk+uu+1][j]);
#endif
            }
            for (; kk < j; ++kk)
              table[i][j] = max_score(table[i][j], table[i][kk] + table[kk+1][j]);
#else
            for (k = i + 1; k < j; k++)
              table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
#endif
          } else {
            /* cold subcase: sampled scan (coarse) */
            /* Only examine every 2nd split point (k += 2). This reduces work but may miss optimum. */
#if defined(LU) && (UNROLL > 1)
            /* with unroll, still sample by step=2 */
            int kk = i + 1;
            for (; kk + (UNROLL - 1) < j; kk += UNROLL) {
              for (int uu = 0; uu < UNROLL; ++uu) {
                int pos = kk + uu;
                if (((pos - (i+1)) % 2) == 0) /* sample only even offsets */
                  table[i][j] = max_score(table[i][j], table[i][pos] + table[pos+1][j]);
              }
            }
            for (; kk < j; ++kk)
              if (((kk - (i+1)) % 2) == 0)
                table[i][j] = max_score(table[i][j], table[i][kk] + table[kk+1][j]);
#else
            for (k = i + 1; k < j; k += 2)
              table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
#endif
            /* Note: we purposely DO NOT rescale sampled results here because max() does not permit a simple scaling compensation.
               Therefore the cold path is an approximation that may under-estimate the true optimum. */
          }
        }
      }
#else
      /* baseline exact scan */
      for (k = i + 1; k < j; k++)
        table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
#endif

    } /* for j */
  } /* for i */
#pragma endscop

}




int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(seq, base, N, n);
  POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_nussinov (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(table)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(seq);
  POLYBENCH_FREE_ARRAY(table);

  return 0;
}
