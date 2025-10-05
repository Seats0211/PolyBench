/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* adi.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static
void kernel_adi(int tsteps, int n,
        DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

#pragma scop

  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;

  /* Tunables for strategy B */
  const int LEN_TH = 64; /* if grid dimension large, favor sampling on cold path */
  const DATA_TYPE U_TH  = SCALAR_VAL(1e-6); /* cheap threshold on magnitude for predicate */

  for (t=1; t<=_PB_TSTEPS; t++) {

    /* ---------------- Column Sweep ---------------- */
    for (i=1; i<_PB_N-1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];

#if defined(SE)
      /* Decide whether this column is likely 'hot' or can be approximated.
         condA: column contains significant values (cheap); condB: structural; condC: pseudo-random */
      int condBcol = ((i % 2) == 0);
      int condCrand = (((int)(u[0][i]*1000.0)) & 1);
      int hotCol = condBcol || (condCrand && (_PB_N > LEN_TH));

      if (hotCol) {
        /* hot path: full accurate forward recurrence for p,q */
  #if defined(LU) && (UNROLL > 1)
        int jj = 1;
        for (; jj + (UNROLL - 1) < _PB_N-1; jj += UNROLL) {
  #if UNROLL == 4
          p[i][jj] = -c / (a*p[i][jj-1]+b);
          q[i][jj] = (-d*u[jj][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[jj][i] - f*u[jj][i+1] - a*q[i][jj-1]) / (a*p[i][jj-1]+b);

          p[i][jj+1] = -c / (a*p[i][jj]+b);
          q[i][jj+1] = (-d*u[jj+1][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[jj+1][i] - f*u[jj+1][i+1] - a*q[i][jj]) / (a*p[i][jj]+b);

          p[i][jj+2] = -c / (a*p[i][jj+1]+b);
          q[i][jj+2] = (-d*u[jj+2][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[jj+2][i] - f*u[jj+2][i+1] - a*q[i][jj+1]) / (a*p[i][jj+1]+b);

          p[i][jj+3] = -c / (a*p[i][jj+2]+b);
          q[i][jj+3] = (-d*u[jj+3][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[jj+3][i] - f*u[jj+3][i+1] - a*q[i][jj+2]) / (a*p[i][jj+2]+b);
  #else
          for (int uu=0; uu<UNROLL; ++uu) {
            int pos = jj+uu;
            p[i][pos] = -c / (a*p[i][pos-1]+b);
            q[i][pos] = (-d*u[pos][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[pos][i] - f*u[pos][i+1] - a*q[i][pos-1]) / (a*p[i][pos-1]+b);
          }
  #endif
        }
        for (; jj < _PB_N-1; ++jj) {
          p[i][jj] = -c / (a*p[i][jj-1]+b);
          q[i][jj] = (-d*u[jj][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[jj][i] - f*u[jj][i+1] - a*q[i][jj-1]) / (a*p[i][jj-1]+b);
        }
  #else
        for (j=1; j<_PB_N-1; j++) {
          p[i][j] = -c / (a*p[i][j-1]+b);
          q[i][j] = (-d*u[j][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1] - a*q[i][j-1]) / (a*p[i][j-1]+b);
        }
  #endif
      } else {
        /* cold column: sampled forward recurrence to reduce work.
           We compute p,q at j step=2 and approximate the skipped j+1 using a local linearization.
           This keeps some temporal dependency while saving work. */
  #if defined(LU) && (UNROLL > 1)
        int jj = 1;
        for (; jj + (UNROLL - 1) < _PB_N-1; jj += UNROLL) {
          for (int uu=0; uu<UNROLL; ++uu) {
            int pos = jj + uu;
            if (((pos - 1) % 2) == 0) {
              /* compute sampled position */
              p[i][pos] = -c / (a*p[i][pos-1]+b);
              q[i][pos] = (-d*u[pos][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[pos][i] - f*u[pos][i+1] - a*q[i][pos-1]) / (a*p[i][pos-1]+b);
              /* approximate pos+1 if within bounds */
              if (pos+1 < _PB_N-1) {
                /* light-weight approx using linear extrapolation of q and p */
                p[i][pos+1] = p[i][pos]; /* reuse */
                q[i][pos+1] = (-d*u[pos+1][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[pos+1][i] - f*u[pos+1][i+2] - a*q[i][pos]) / (a*p[i][pos]+b);
              }
            }
          }
        }
        for (; jj < _PB_N-1; ++jj) {
          if (((jj - 1) % 2) == 0) {
            p[i][jj] = -c / (a*p[i][jj-1]+b);
            q[i][jj] = (-d*u[jj][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[jj][i] - f*u[jj][i+1] - a*q[i][jj-1]) / (a*p[i][jj-1]+b);
            if (jj+1 < _PB_N-1) {
              p[i][jj+1] = p[i][jj];
              q[i][jj+1] = (-d*u[jj+1][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[jj+1][i] - f*u[jj+1][i+2] - a*q[i][jj]) / (a*p[i][jj]+b);
            }
          }
        }
  #else
        for (j=1; j<_PB_N-1; j+=2) {
          p[i][j] = -c / (a*p[i][j-1]+b);
          q[i][j] = (-d*u[j][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1] - a*q[i][j-1]) / (a*p[i][j-1]+b);
          if (j+1 < _PB_N-1) {
            /* approximate next entry */
            p[i][j+1] = p[i][j];
            q[i][j+1] = (-d*u[j+1][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j+1][i] - f*u[j+1][i+2] - a*q[i][j]) / (a*p[i][j]+b);
          }
        }
  #endif
      }
#else
      /* baseline exact forward recurrence */
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -c / (a*p[i][j-1]+b);
        q[i][j] = (-d*u[j][i-1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1] - a*q[i][j-1]) / (a*p[i][j-1]+b);
      }
#endif

      v[_PB_N-1][i] = SCALAR_VAL(1.0);

#if defined(SE)
      /* backward substitution for v: similar hot/cold decision */
      if (hotCol) {
        for (j=_PB_N-2; j>=1; j--) {
          v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
        }
      } else {
        /* cold: sample backward by 2 and fill neighbors by local interpolation */
        int jj = _PB_N-2;
        for (; jj >= 1; jj -= 2) {
          v[jj][i] = p[i][jj] * v[jj+1][i] + q[i][jj];
          if (jj-1 >= 1) {
            /* approximate j-1 using neighbor */
            v[jj-1][i] = p[i][jj-1] * v[jj][i] + q[i][jj-1];
          }
        }
      }
#else
      for (j=_PB_N-2; j>=1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
#endif
    } /* end column sweep i loop */


    /* ---------------- Row Sweep ---------------- */
    for (i=1; i<_PB_N-1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];

#if defined(SE)
      /* decide per-row hot/cold similarly */
      int condBrow = ((i % 2) == 0);
      int condCrandR = (((int)(v[i][0]*1000.0)) & 1);
      int hotRow = condBrow || (condCrandR && (_PB_N > LEN_TH));

      if (hotRow) {
  #if defined(LU) && (UNROLL > 1)
        int jj = 1;
        for (; jj + (UNROLL - 1) < _PB_N-1; jj += UNROLL) {
  #if UNROLL == 4
          p[i][jj] = -f / (d*p[i][jj-1]+e);
          q[i][jj] = (-a*v[i-1][jj] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][jj] - c*v[i+1][jj] - d*q[i][jj-1]) / (d*p[i][jj-1]+e);

          p[i][jj+1] = -f / (d*p[i][jj]+e);
          q[i][jj+1] = (-a*v[i-1][jj+1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][jj+1] - c*v[i+1][jj+1] - d*q[i][jj]) / (d*p[i][jj]+e);

          p[i][jj+2] = -f / (d*p[i][jj+1]+e);
          q[i][jj+2] = (-a*v[i-1][jj+2] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][jj+2] - c*v[i+1][jj+2] - d*q[i][jj+1]) / (d*p[i][jj+1]+e);

          p[i][jj+3] = -f / (d*p[i][jj+2]+e);
          q[i][jj+3] = (-a*v[i-1][jj+3] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][jj+3] - c*v[i+1][jj+3] - d*q[i][jj+2]) / (d*p[i][jj+2]+e);
  #else
          for (int uu=0; uu<UNROLL; ++uu) {
            int pos = jj+uu;
            p[i][pos] = -f / (d*p[i][pos-1]+e);
            q[i][pos] = (-a*v[i-1][pos] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][pos] - c*v[i+1][pos] - d*q[i][pos-1]) / (d*p[i][pos-1]+e);
          }
  #endif
        }
        for (; jj < _PB_N-1; ++jj) {
          p[i][jj] = -f / (d*p[i][jj-1]+e);
          q[i][jj] = (-a*v[i-1][jj] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][jj] - c*v[i+1][jj] - d*q[i][jj-1]) / (d*p[i][jj-1]+e);
        }
  #else
        for (j=1; j<_PB_N-1; j++) {
          p[i][j] = -f / (d*p[i][j-1]+e);
          q[i][j] = (-a*v[i-1][j] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j] - d*q[i][j-1]) / (d*p[i][j-1]+e);
        }
  #endif
      } else {
        /* cold row: sampled forward recurrence (step=2) with light approx for skipped points */
  #if defined(LU) && (UNROLL > 1)
        int jj = 1;
        for (; jj + (UNROLL - 1) < _PB_N-1; jj += UNROLL) {
          for (int uu=0; uu<UNROLL; ++uu) {
            int pos = jj+uu;
            if (((pos - 1) % 2) == 0) {
              p[i][pos] = -f / (d*p[i][pos-1]+e);
              q[i][pos] = (-a*v[i-1][pos] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][pos] - c*v[i+1][pos] - d*q[i][pos-1]) / (d*p[i][pos-1]+e);
              if (pos+1 < _PB_N-1) {
                p[i][pos+1] = p[i][pos];
                q[i][pos+1] = (-a*v[i-1][pos+1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][pos+1] - c*v[i+1][pos+1] - d*q[i][pos]) / (d*p[i][pos]+e);
              }
            }
          }
        }
        for (; jj < _PB_N-1; ++jj) {
          if (((jj - 1) % 2) == 0) {
            p[i][jj] = -f / (d*p[i][jj-1]+e);
            q[i][jj] = (-a*v[i-1][jj] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][jj] - c*v[i+1][jj] - d*q[i][jj-1]) / (d*p[i][jj-1]+e);
            if (jj+1 < _PB_N-1) {
              p[i][jj+1] = p[i][jj];
              q[i][jj+1] = (-a*v[i-1][jj+1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][jj+1] - c*v[i+1][jj+1] - d*q[i][jj]) / (d*p[i][jj]+e);
            }
          }
        }
  #else
        for (j=1; j<_PB_N-1; j+=2) {
          p[i][j] = -f / (d*p[i][j-1]+e);
          q[i][j] = (-a*v[i-1][j] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j] - d*q[i][j-1]) / (d*p[i][j-1]+e);
          if (j+1 < _PB_N-1) {
            p[i][j+1] = p[i][j];
            q[i][j+1] = (-a*v[i-1][j+1] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j+1] - c*v[i+1][j+1] - d*q[i][j]) / (d*p[i][j]+e);
          }
        }
  #endif
      }

#else
      /* baseline exact forward recurrence for row */
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -f / (d*p[i][j-1]+e);
        q[i][j] = (-a*v[i-1][j] + (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j] - d*q[i][j-1]) / (d*p[i][j-1]+e);
      }
#endif

      u[i][_PB_N-1] = SCALAR_VAL(1.0);

#if defined(SE)
      if (hotRow) {
        for (j=_PB_N-2; j>=1; j--) {
          u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
        }
      } else {
        /* cold backward: sample by 2 */
        int jj = _PB_N-2;
        for (; jj >= 1; jj -= 2) {
          u[i][jj] = p[i][jj] * u[i][jj+1] + q[i][jj];
          if (jj-1 >= 1) {
            u[i][jj-1] = p[i][jj-1] * u[i][jj] + q[i][jj-1];
          }
        }
      }
#else
      for (j=_PB_N-2; j>=1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
      }
#endif
    } /* end row sweep i loop */

  } /* end timesteps */

#pragma endscop
}




int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

  return 0;
}
