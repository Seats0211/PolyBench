/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 2mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm(int ni, int nj, int nk, int nl,
        DATA_TYPE alpha,
        DATA_TYPE beta,
        DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
        DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
        DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* D := alpha*A*B*C + beta*D */
  /* First phase: tmp = alpha * A * B  (tmp is NI x NJ) */
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++) {
#if defined(SE)
      const DATA_TYPE ALPHA_TH = SCALAR_VAL(1e-3);
      const DATA_TYPE N_TH = SCALAR_VAL(64.0);
      int condA = (fabs(alpha) > ALPHA_TH);
      int condB = ((i + j) % 2 == 0);
      int condC = (((int)(A[i][0] * 1000.0)) & 1);

      DATA_TYPE acc = SCALAR_VAL(0.0);

      if (condA) {
        if (condB || (condC && (nk > N_TH))) {
          /* hot: full accumulation, optional unroll */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NK; kk += UNROLL) {
#if UNROLL == 4
            acc += alpha * A[i][kk+0] * B[kk+0][j];
            acc += alpha * A[i][kk+1] * B[kk+1][j];
            acc += alpha * A[i][kk+2] * B[kk+2][j];
            acc += alpha * A[i][kk+3] * B[kk+3][j];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              acc += alpha * A[i][kk+uu] * B[kk+uu][j];
#endif
          }
          for (; kk < _PB_NK; ++kk)
            acc += alpha * A[i][kk] * B[kk][j];
#else
          for (k = 0; k < _PB_NK; ++k)
            acc += alpha * A[i][k] * B[k][j];
#endif
        } else {
          /* condA true but subguard false -> reduced-cost approximate: sample k+=2 */
          for (k = 0; k < _PB_NK; k += 2)
            acc += alpha * A[i][k] * B[k][j];
          acc *= SCALAR_VAL(2.0); /* simple compensation */
        }
      } else {
        /* condA false: alternate branching */
        if (condC && (nk > N_TH)) {
          /* alternate hot: full accumulation */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NK; kk += UNROLL) {
#if UNROLL == 4
            acc += alpha * A[i][kk+0] * B[kk+0][j];
            acc += alpha * A[i][kk+1] * B[kk+1][j];
            acc += alpha * A[i][kk+2] * B[kk+2][j];
            acc += alpha * A[i][kk+3] * B[kk+3][j];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              acc += alpha * A[i][kk+uu] * B[kk+uu][j];
#endif
          }
          for (; kk < _PB_NK; ++kk)
            acc += alpha * A[i][kk] * B[kk][j];
#else
          for (k = 0; k < _PB_NK; ++k)
            acc += alpha * A[i][k] * B[k][j];
#endif
        } else {
          /* cold path: approximate sampling */
          for (k = 0; k < _PB_NK; k += 2)
            acc += alpha * A[i][k] * B[k][j];
          acc *= SCALAR_VAL(2.0);
        }
      }

      tmp[i][j] = acc;
#else
      /* baseline */
      tmp[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_NK; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
#endif
    }
  }

  /* Second phase: D = beta * D + tmp * C  (D is NI x NL) */
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NL; j++) {
#if defined(SE)
      const DATA_TYPE BETA_TH = SCALAR_VAL(1e-3);
      const DATA_TYPE N_TH2 = SCALAR_VAL(64.0);
      int condA2 = (fabs(beta) > BETA_TH);
      int condB2 = ((i + j) % 2 == 0);
      int condC2 = (((int)(tmp[i][0] * 1000.0)) & 1);

      DATA_TYPE acc2 = SCALAR_VAL(0.0);

      if (condA2) {
        if (condB2 || (condC2 && (nj > N_TH2))) {
          /* hot: full accumulation */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NJ; kk += UNROLL) {
#if UNROLL == 4
            acc2 += tmp[i][kk+0] * C[kk+0][j];
            acc2 += tmp[i][kk+1] * C[kk+1][j];
            acc2 += tmp[i][kk+2] * C[kk+2][j];
            acc2 += tmp[i][kk+3] * C[kk+3][j];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              acc2 += tmp[i][kk+uu] * C[kk+uu][j];
#endif
          }
          for (; kk < _PB_NJ; ++kk)
            acc2 += tmp[i][kk] * C[kk][j];
#else
          for (k = 0; k < _PB_NJ; ++k)
            acc2 += tmp[i][k] * C[k][j];
#endif
        } else {
          /* reduced-cost approximate */
          for (k = 0; k < _PB_NJ; k += 2)
            acc2 += tmp[i][k] * C[k][j];
          acc2 *= SCALAR_VAL(2.0);
        }
      } else {
        if (condC2 && (nj > N_TH2)) {
          /* alternate hot */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NJ; kk += UNROLL) {
#if UNROLL == 4
            acc2 += tmp[i][kk+0] * C[kk+0][j];
            acc2 += tmp[i][kk+1] * C[kk+1][j];
            acc2 += tmp[i][kk+2] * C[kk+2][j];
            acc2 += tmp[i][kk+3] * C[kk+3][j];
#else
            for (int uu = 0; uu < UNROLL; ++uu)
              acc2 += tmp[i][kk+uu] * C[kk+uu][j];
#endif
          }
          for (; kk < _PB_NJ; ++kk)
            acc2 += tmp[i][kk] * C[kk][j];
#else
          for (k = 0; k < _PB_NJ; ++k)
            acc2 += tmp[i][k] * C[k][j];
#endif
        } else {
          /* cold path */
          for (k = 0; k < _PB_NJ; k += 2)
            acc2 += tmp[i][k] * C[k][j];
          acc2 *= SCALAR_VAL(2.0);
        }
      }

      D[i][j] = beta * D[i][j] + acc2;
#else
      /* baseline */
      D[i][j] *= beta;
      for (k = 0; k < _PB_NJ; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
#endif
    }
  }
#pragma endscop

}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_2mm (ni, nj, nk, nl,
	      alpha, beta,
	      POLYBENCH_ARRAY(tmp),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  return 0;
}
