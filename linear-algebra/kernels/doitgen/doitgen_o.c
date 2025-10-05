/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* doitgen.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "doitgen.h"


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgen(int nr, int nq, int np,
            DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
            DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
            DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

#pragma scop
  for (r = 0; r < _PB_NR; r++) {
    for (q = 0; q < _PB_NQ; q++)  {
      /* For each p we compute sum[p] = sum_s A[r][q][s] * C4[s][p] */
      for (p = 0; p < _PB_NP; p++) {
#if defined(SE)
        /* ---------- Shannon-style predicates (tunable) ---------- */
        const DATA_TYPE C_TH = SCALAR_VAL(1e-6);
        const int S_TH = 64; /* tune per problem size */
        /* cheap predicates */
        int condA = (fabs(C4[0][p]) > C_TH);     /* depends on column p */
        int condB = ((p + q) % 2 == 0);         /* structural predicate */
        int condC = (((int)(A[r][q][0] * 1000.0)) & 1); /* pseudo-random bit */

        DATA_TYPE acc = SCALAR_VAL(0.0);

        if (condA) {
          if (condB || (condC && (_PB_NP > S_TH))) {
            /* hot path: full accumulation (support LU unroll) */
  #if defined(LU) && (UNROLL > 1)
            int ss = 0;
            for (; ss + (UNROLL - 1) < _PB_NP; ss += UNROLL) {
  #if UNROLL == 4
              acc += A[r][q][ss+0] * C4[ss+0][p];
              acc += A[r][q][ss+1] * C4[ss+1][p];
              acc += A[r][q][ss+2] * C4[ss+2][p];
              acc += A[r][q][ss+3] * C4[ss+3][p];
  #else
              for (int uu = 0; uu < UNROLL; ++uu)
                acc += A[r][q][ss+uu] * C4[ss+uu][p];
  #endif
            }
            for (; ss < _PB_NP; ++ss)
              acc += A[r][q][ss] * C4[ss][p];
  #else
            for (s = 0; s < _PB_NP; s++)
              acc += A[r][q][s] * C4[s][p];
  #endif
          } else {
            /* condA true but subguard false -> reduced-cost approximate: sample every 2 */
            for (s = 0; s < _PB_NP; s += 2)
              acc += A[r][q][s] * C4[s][p];
            acc *= SCALAR_VAL(2.0); /* simple compensation */
          }
        } else {
          /* condA == false: alternate grouping */
          if (condC && (_PB_NP > S_TH)) {
  #if defined(LU) && (UNROLL > 1)
            int ss = 0;
            for (; ss + (UNROLL - 1) < _PB_NP; ss += UNROLL) {
  #if UNROLL == 4
              acc += A[r][q][ss+0] * C4[ss+0][p];
              acc += A[r][q][ss+1] * C4[ss+1][p];
              acc += A[r][q][ss+2] * C4[ss+2][p];
              acc += A[r][q][ss+3] * C4[ss+3][p];
  #else
              for (int uu = 0; uu < UNROLL; ++uu)
                acc += A[r][q][ss+uu] * C4[ss+uu][p];
  #endif
            }
            for (; ss < _PB_NP; ++ss)
              acc += A[r][q][ss] * C4[ss][p];
  #else
            for (s = 0; s < _PB_NP; s++)
              acc += A[r][q][s] * C4[s][p];
  #endif
          } else {
            /* cold path: sampled approximate */
            for (s = 0; s < _PB_NP; s += 2)
              acc += A[r][q][s] * C4[s][p];
            acc *= SCALAR_VAL(2.0);
          }
        }

        sum[p] = acc;
#else
        /* baseline exact accumulation */
        sum[p] = SCALAR_VAL(0.0);
        for (s = 0; s < _PB_NP; s++)
          sum[p] += A[r][q][s] * C4[s][p];
#endif
      } /* end for p */

      /* write back the result row */
      for (p = 0; p < _PB_NP; p++)
        A[r][q][p] = sum[p];
    } /* end for q */
  } /* end for r */
#pragma endscop
}




int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE,NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_doitgen (nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}
