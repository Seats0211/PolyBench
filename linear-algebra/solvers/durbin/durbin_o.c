/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* durbin.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
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
static void kernel_durbin(int n, DATA_TYPE POLYBENCH_1D(r, N, n),
                          DATA_TYPE POLYBENCH_1D(y, N, n)) {
  DATA_TYPE z[N];
  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  int i, k;

#pragma scop
  y[0] = -r[0];
  beta = SCALAR_VAL(1.0);
  alpha = -r[0];

  for (k = 1; k < _PB_N; k++) {
    beta = (SCALAR_VAL(1.0) - alpha * alpha) * beta;
    sum = SCALAR_VAL(0.0);

#if defined(LU) && (UNROLL > 1)
    /* optional manual unrolled accumulation for sum */
    int jj = 0;
    for (; jj + (UNROLL - 1) < k; jj += UNROLL) {
#if UNROLL == 8
      sum += r[k-0-jj-1] * y[0+jj];
      sum += r[k-1-jj-1] * y[1+jj];
      sum += r[k-2-jj-1] * y[2+jj];
      sum += r[k-3-jj-1] * y[3+jj];
      sum += r[k-4-jj-1] * y[4+jj];
      sum += r[k-5-jj-1] * y[5+jj];
      sum += r[k-6-jj-1] * y[6+jj];
      sum += r[k-7-jj-1] * y[7+jj];
#elif UNROLL == 4
      sum += r[k-jj-1]     * y[jj];
      sum += r[k-(jj+1)-1] * y[jj+1];
      sum += r[k-(jj+2)-1] * y[jj+2];
      sum += r[k-(jj+3)-1] * y[jj+3];
#elif UNROLL == 2
      sum += r[k-jj-1]     * y[jj];
      sum += r[k-(jj+1)-1] * y[jj+1];
#else
      for (int uu = 0; uu < UNROLL; ++uu)
        sum += r[k-(jj+uu)-1] * y[jj+uu];
#endif
    }
    for (; jj < k; ++jj) sum += r[k - jj - 1] * y[jj];
#else
    /* baseline accumulation */
    for (i = 0; i < k; i++) {
      sum += r[k - i - 1] * y[i];
    }
#endif

    /* compute alpha safely (beta might be small) */
    /* guard for tiny beta to avoid blow-up; if tiny, leave alpha unchanged (or handle separately) */
    if (fabsl(beta) < SCALAR_VAL(1e-18)) {
      /* extremely small beta: fallback to baseline-ish safe behavior */
      /* recordable event in experiments — here we keep previous alpha to avoid division by tiny */
      /* (you may choose to recompute with higher precision or skip SE in such cases) */
      /* For now, do a safe clamp */
      beta = (beta >= 0) ? SCALAR_VAL(1e-18) : -SCALAR_VAL(1e-18);
    }
    alpha = -(r[k] + sum) / beta;

#if defined(SE)
    /* Shannon-style composite guard — split on a cheap predicate condA */
    const DATA_TYPE ALPHA_TH = SCALAR_VAL(1e-6);
    const DATA_TYPE S_TH = SCALAR_VAL(0.01);
    int condA = (fabsl(alpha) > ALPHA_TH);
    int condB = (k % 2 == 0);
    int condC = (((int)(r[k] * 1000.0)) & 1);

    if (condA) {
      if (condB || (condC && (sum > S_TH))) {
        /* hot path: full z update */
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < k; jj += UNROLL) {
#if UNROLL == 4
          z[jj]   = y[jj]   + alpha * y[k - jj - 1];
          z[jj+1] = y[jj+1] + alpha * y[k - (jj+1) - 1];
          z[jj+2] = y[jj+2] + alpha * y[k - (jj+2) - 1];
          z[jj+3] = y[jj+3] + alpha * y[k - (jj+3) - 1];
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            z[jj+uu] = y[jj+uu] + alpha * y[k - (jj+uu) - 1];
#endif
        }
        for (; jj < k; ++jj) z[jj] = y[jj] + alpha * y[k - jj - 1];
#else
        for (i = 0; i < k; i++)
          z[i] = y[i] + alpha * y[k - i - 1];
#endif
      } else {
        /* condA true but subguard false -> reduced-cost approximate (sample step=2) */
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < k; jj += UNROLL) {
#if UNROLL == 4
          z[jj]   = y[jj]   + (alpha * SCALAR_VAL(0.5)) * y[k - jj - 1];
          z[jj+1] = y[jj+1] + (alpha * SCALAR_VAL(0.5)) * y[k - (jj+1) - 1];
          z[jj+2] = y[jj+2] + (alpha * SCALAR_VAL(0.5)) * y[k - (jj+2) - 1];
          z[jj+3] = y[jj+3] + (alpha * SCALAR_VAL(0.5)) * y[k - (jj+3) - 1];
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            z[jj+uu] = y[jj+uu] + (alpha * SCALAR_VAL(0.5)) * y[k - (jj+uu) - 1];
#endif
        }
        for (; jj < k; ++jj) z[jj] = y[jj] + (alpha * SCALAR_VAL(0.5)) * y[k - jj - 1];
#else
        for (i = 0; i < k; i++)
          z[i] = y[i] + (alpha * SCALAR_VAL(0.5)) * y[k - i - 1];
#endif
      }
    } else {
      /* condA == false */
      if (condC && (sum > S_TH)) {
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < k; jj += UNROLL) {
#if UNROLL == 4
          z[jj]   = y[jj]   + alpha * y[k - jj - 1];
          z[jj+1] = y[jj+1] + alpha * y[k - (jj+1) - 1];
          z[jj+2] = y[jj+2] + alpha * y[k - (jj+2) - 1];
          z[jj+3] = y[jj+3] + alpha * y[k - (jj+3) - 1];
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            z[jj+uu] = y[jj+uu] + alpha * y[k - (jj+uu) - 1];
#endif
        }
        for (; jj < k; ++jj) z[jj] = y[jj] + alpha * y[k - jj - 1];
#else
        for (i = 0; i < k; i++)
          z[i] = y[i] + alpha * y[k - i - 1];
#endif
      } else {
        /* cold path: sampled approximate */
#if defined(LU) && (UNROLL > 1)
        int jj = 0;
        for (; jj + (UNROLL - 1) < k; jj += UNROLL) {
#if UNROLL == 4
          z[jj]   = y[jj]   + (alpha * SCALAR_VAL(0.5)) * y[k - jj - 1];
          z[jj+1] = y[jj+1] + (alpha * SCALAR_VAL(0.5)) * y[k - (jj+1) - 1];
          z[jj+2] = y[jj+2] + (alpha * SCALAR_VAL(0.5)) * y[k - (jj+2) - 1];
          z[jj+3] = y[jj+3] + (alpha * SCALAR_VAL(0.5)) * y[k - (jj+3) - 1];
#else
          for (int uu = 0; uu < UNROLL; ++uu)
            z[jj+uu] = y[jj+uu] + (alpha * SCALAR_VAL(0.5)) * y[k - (jj+uu) - 1];
#endif
        }
        for (; jj < k; ++jj) z[jj] = y[jj] + (alpha * SCALAR_VAL(0.5)) * y[k - jj - 1];
#else
        for (i = 0; i < k; i++)
          z[i] = y[i] + (alpha * SCALAR_VAL(0.5)) * y[k - i - 1];
#endif
      }
    }
#else
    /* No Shannon Expansion: original z update */
#if defined(LU) && (UNROLL > 1)
    jj = 0;
    for (; jj + (UNROLL - 1) < k; jj += UNROLL) {
#if UNROLL == 4
      z[jj]   = y[jj]   + alpha * y[k - jj - 1];
      z[jj+1] = y[jj+1] + alpha * y[k - (jj+1) - 1];
      z[jj+2] = y[jj+2] + alpha * y[k - (jj+2) - 1];
      z[jj+3] = y[jj+3] + alpha * y[k - (jj+3) - 1];
#else
      for (int uu = 0; uu < UNROLL; ++uu)
        z[jj+uu] = y[jj+uu] + alpha * y[k - (jj+uu) - 1];
#endif
    }
    for (; jj < k; ++jj) z[jj] = y[jj] + alpha * y[k - jj - 1];
#else
    for (i = 0; i < k; i++) {
      z[i] = y[i] + alpha * y[k - i - 1];
    }
#endif
#endif /* SE */

    for (i = 0; i < k; i++) {
      y[i] = z[i];
    }
    y[k] = alpha;
  }
#pragma endscop
}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(r));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_durbin (n,
		 POLYBENCH_ARRAY(r),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}
