/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 3mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
        DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
        DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
        DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
        DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
        DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* E := A * B  (NI x NJ) */
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++) {
#if defined(SE)
      const DATA_TYPE TH1 = SCALAR_VAL(1e-3);
      const DATA_TYPE KTH = SCALAR_VAL(64.0); /* 可根据 problem size 调整 */
      int condA = (fabs(A[i][0]) > TH1);           /* cheap predicate */
      int condB = ((i + j) % 2 == 0);
      int condC = (((int)(B[0][j] * 1000.0)) & 1);

      DATA_TYPE accE = SCALAR_VAL(0.0);
      if (condA) {
        if (condB || (condC && (nk > KTH))) {
          /* 热路径：完整累加（支持 LU 展开） */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NK; kk += UNROLL) {
#if UNROLL == 4
            accE += A[i][kk+0] * B[kk+0][j];
            accE += A[i][kk+1] * B[kk+1][j];
            accE += A[i][kk+2] * B[kk+2][j];
            accE += A[i][kk+3] * B[kk+3][j];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              accE += A[i][kk+uu] * B[kk+uu][j];
#endif
          }
          for (; kk < _PB_NK; ++kk) accE += A[i][kk] * B[kk][j];
#else
          for (k = 0; k < _PB_NK; ++k) accE += A[i][k] * B[k][j];
#endif
        } else {
          /* condA true but subguard false -> 近似路径：k += 2 采样并放大 */
          for (k = 0; k < _PB_NK; k += 2) accE += A[i][k] * B[k][j];
          accE *= SCALAR_VAL(2.0);
        }
      } else {
        if (condC && (nk > KTH)) {
          /* 另一条热路径：完整 */
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NK; kk += UNROLL) {
#if UNROLL == 4
            accE += A[i][kk+0] * B[kk+0][j];
            accE += A[i][kk+1] * B[kk+1][j];
            accE += A[i][kk+2] * B[kk+2][j];
            accE += A[i][kk+3] * B[kk+3][j];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              accE += A[i][kk+uu] * B[kk+uu][j];
#endif
          }
          for (; kk < _PB_NK; ++kk) accE += A[i][kk] * B[kk][j];
#else
          for (k = 0; k < _PB_NK; ++k) accE += A[i][k] * B[k][j];
#endif
        } else {
          /* 冷路径：采样近似 */
          for (k = 0; k < _PB_NK; k += 2) accE += A[i][k] * B[k][j];
          accE *= SCALAR_VAL(2.0);
        }
      }
      E[i][j] = accE;
#else
      /* baseline */
      E[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_NK; ++k)
        E[i][j] += A[i][k] * B[k][j];
#endif
    }
  }

  /* F := C * D  (NJ x NL) */
  for (i = 0; i < _PB_NJ; i++) {
    for (j = 0; j < _PB_NL; j++) {
#if defined(SE)
      const DATA_TYPE TH2 = SCALAR_VAL(1e-3);
      const DATA_TYPE KTH2 = SCALAR_VAL(64.0);
      int condA2 = (fabs(C[i][0]) > TH2);
      int condB2 = ((i + j) % 2 == 0);
      int condC2 = (((int)(D[0][j] * 1000.0)) & 1);

      DATA_TYPE accF = SCALAR_VAL(0.0);
      if (condA2) {
        if (condB2 || (condC2 && (nm > KTH2))) {
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NM; kk += UNROLL) {
#if UNROLL == 4
            accF += C[i][kk+0] * D[kk+0][j];
            accF += C[i][kk+1] * D[kk+1][j];
            accF += C[i][kk+2] * D[kk+2][j];
            accF += C[i][kk+3] * D[kk+3][j];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              accF += C[i][kk+uu] * D[kk+uu][j];
#endif
          }
          for (; kk < _PB_NM; ++kk) accF += C[i][kk] * D[kk][j];
#else
          for (k = 0; k < _PB_NM; ++k) accF += C[i][k] * D[k][j];
#endif
        } else {
          for (k = 0; k < _PB_NM; k += 2) accF += C[i][k] * D[k][j];
          accF *= SCALAR_VAL(2.0);
        }
      } else {
        if (condC2 && (nm > KTH2)) {
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NM; kk += UNROLL) {
#if UNROLL == 4
            accF += C[i][kk+0] * D[kk+0][j];
            accF += C[i][kk+1] * D[kk+1][j];
            accF += C[i][kk+2] * D[kk+2][j];
            accF += C[i][kk+3] * D[kk+3][j];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              accF += C[i][kk+uu] * D[kk+uu][j];
#endif
          }
          for (; kk < _PB_NM; ++kk) accF += C[i][kk] * D[kk][j];
#else
          for (k = 0; k < _PB_NM; ++k) accF += C[i][k] * D[k][j];
#endif
        } else {
          for (k = 0; k < _PB_NM; k += 2) accF += C[i][k] * D[k][j];
          accF *= SCALAR_VAL(2.0);
        }
      }
      F[i][j] = accF;
#else
      /* baseline */
      F[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_NM; ++k)
        F[i][j] += C[i][k] * D[k][j];
#endif
    }
  }

  /* G := E * F  (NI x NL) */
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NL; j++) {
#if defined(SE)
      const DATA_TYPE TH3 = SCALAR_VAL(1e-3);
      const DATA_TYPE KTH3 = SCALAR_VAL(64.0);
      int condA3 = (fabs(E[i][0]) > TH3);
      int condB3 = ((i + j) % 2 == 0);
      int condC3 = (((int)(F[0][j] * 1000.0)) & 1);

      DATA_TYPE accG = SCALAR_VAL(0.0);
      if (condA3) {
        if (condB3 || (condC3 && (nj > KTH3))) {
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NJ; kk += UNROLL) {
#if UNROLL == 4
            accG += E[i][kk+0] * F[kk+0][j];
            accG += E[i][kk+1] * F[kk+1][j];
            accG += E[i][kk+2] * F[kk+2][j];
            accG += E[i][kk+3] * F[kk+3][j];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              accG += E[i][kk+uu] * F[kk+uu][j];
#endif
          }
          for (; kk < _PB_NJ; ++kk) accG += E[i][kk] * F[kk][j];
#else
          for (k = 0; k < _PB_NJ; ++k) accG += E[i][k] * F[k][j];
#endif
        } else {
          for (k = 0; k < _PB_NJ; k += 2) accG += E[i][k] * F[k][j];
          accG *= SCALAR_VAL(2.0);
        }
      } else {
        if (condC3 && (nj > KTH3)) {
#if defined(LU) && (UNROLL > 1)
          int kk = 0;
          for (; kk + (UNROLL - 1) < _PB_NJ; kk += UNROLL) {
#if UNROLL == 4
            accG += E[i][kk+0] * F[kk+0][j];
            accG += E[i][kk+1] * F[kk+1][j];
            accG += E[i][kk+2] * F[kk+2][j];
            accG += E[i][kk+3] * F[kk+3][j];
#else
            for (int uu=0; uu<UNROLL; ++uu)
              accG += E[i][kk+uu] * F[kk+uu][j];
#endif
          }
          for (; kk < _PB_NJ; ++kk) accG += E[i][kk] * F[kk][j];
#else
          for (k = 0; k < _PB_NJ; ++k) accG += E[i][k] * F[k][j];
#endif
        } else {
          for (k = 0; k < _PB_NJ; k += 2) accG += E[i][k] * F[k][j];
          accG *= SCALAR_VAL(2.0);
        }
      }
      G[i][j] = accG;
#else
      /* baseline */
      G[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_NJ; ++k)
        G[i][j] += E[i][k] * F[k][j];
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
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_3mm (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(E),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(F),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D),
	      POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  return 0;
}
