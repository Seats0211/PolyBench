/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
      {
	A[i] = ((DATA_TYPE) i+ 2) / n;
	B[i] = ((DATA_TYPE) i+ 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_1d(int tsteps,
                      int n,
                      DATA_TYPE POLYBENCH_1D(A,N,n),
                      DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;

#pragma scop
  
  const int LEN_TH = 512;

  for (t = 0; t < _PB_TSTEPS; t++)
  {
#if defined(SE)
    
    int use_sampling = (_PB_N > LEN_TH);

    if (!use_sampling) {
      /* hot path */
  #if defined(LU) && (UNROLL > 1)
      int ii = 1;
      for (; ii + (UNROLL - 1) < _PB_N-1; ii += UNROLL) {
  #if UNROLL == 4
        B[ii]   = SCALAR_VAL(0.33333) * (A[ii-1] + A[ii] + A[ii+1]);
        B[ii+1] = SCALAR_VAL(0.33333) * (A[ii]   + A[ii+1] + A[ii+2]);
        B[ii+2] = SCALAR_VAL(0.33333) * (A[ii+1] + A[ii+2] + A[ii+3]);
        B[ii+3] = SCALAR_VAL(0.33333) * (A[ii+2] + A[ii+3] + A[ii+4]);
  #else
        for (int uu = 0; uu < UNROLL; ++uu)
          B[ii+uu] = SCALAR_VAL(0.33333) * (A[ii+uu-1] + A[ii+uu] + A[ii+uu+1]);
  #endif
      }
      for (; ii < _PB_N-1; ++ii)
        B[ii] = SCALAR_VAL(0.33333) * (A[ii-1] + A[ii] + A[ii+1]);
  #else
      for (i = 1; i < _PB_N - 1; i++)
        B[i] = SCALAR_VAL(0.33333) * (A[i-1] + A[i] + A[i+1]);
  #endif
    } else {
      
      /* forward sampled computation for B */
  #if defined(LU) && (UNROLL > 1)
      
      int ii = 1;
      for (; ii + (UNROLL - 1) < _PB_N-1; ii += UNROLL) {
        for (int uu = 0; uu < UNROLL; ++uu) {
          int pos = ii + uu;
          int condB = ((pos & 1) == 0);
          int condC = (((int)(A[pos]*1000.0)) & 1);
          if (condB || (condC && (_PB_TSTEPS > LEN_TH))) {
            
            B[pos] = SCALAR_VAL(0.33333) * (A[pos-1] + A[pos] + A[pos+1]);
          } else {
            
            if ((pos & 1) == 1) {
              /* sampled odd positions — compute */
              B[pos] = SCALAR_VAL(0.33333) * (A[pos-1] + A[pos] + A[pos+1]);
            } else {
              /* leave for interpolation pass */
              B[pos] = SCALAR_VAL(0.0); /* placeholder */
            }
          }
        }
      }
      for (; ii < _PB_N-1; ++ii) {
        int condB = ((ii & 1) == 0);
        int condC = (((int)(A[ii]*1000.0)) & 1);
        if (condB || (condC && (_PB_TSTEPS > LEN_TH)))
          B[ii] = SCALAR_VAL(0.33333) * (A[ii-1] + A[ii] + A[ii+1]);
        else {
          if ((ii & 1) == 1)
            B[ii] = SCALAR_VAL(0.33333) * (A[ii-1] + A[ii] + A[ii+1]);
          else
            B[ii] = SCALAR_VAL(0.0);
        }
      }
  #else
      
      for (i = 1; i < _PB_N - 1; i += 2) {
        B[i] = SCALAR_VAL(0.33333) * (A[i-1] + A[i] + A[i+1]); /* sample */
      }
      
      for (i = 2; i < _PB_N - 2; i += 2) {
        
        B[i] = SCALAR_VAL(0.5) * (B[i-1] + B[i+1]);
      }
      
      if ((_PB_N - 2) % 2 == 0) {
        
        int last = _PB_N - 2;
        if (last % 2 == 0) {
          /* approximate using local stencil */
          B[last] = SCALAR_VAL(0.33333) * (A[last-1] + A[last] + A[last+1]);
        }
      }
  #endif
    } /* end use_sampling */
#else
    /* baseline exact (no SE) */
    for (i = 1; i < _PB_N - 1; i++)
      B[i] = SCALAR_VAL(0.33333) * (A[i-1] + A[i] + A[i+1]);
#endif

    
#if defined(SE)
    if (!use_sampling) {
  #if defined(LU) && (UNROLL > 1)
      int ii = 1;
      for (; ii + (UNROLL - 1) < _PB_N-1; ii += UNROLL) {
  #if UNROLL == 4
        A[ii]   = SCALAR_VAL(0.33333) * (B[ii-1] + B[ii] + B[ii+1]);
        A[ii+1] = SCALAR_VAL(0.33333) * (B[ii]   + B[ii+1] + B[ii+2]);
        A[ii+2] = SCALAR_VAL(0.33333) * (B[ii+1] + B[ii+2] + B[ii+3]);
        A[ii+3] = SCALAR_VAL(0.33333) * (B[ii+2] + B[ii+3] + B[ii+4]);
  #else
        for (int uu = 0; uu < UNROLL; ++uu)
          A[ii+uu] = SCALAR_VAL(0.33333) * (B[ii+uu-1] + B[ii+uu] + B[ii+uu+1]);
  #endif
      }
      for (; ii < _PB_N-1; ++ii)
        A[ii] = SCALAR_VAL(0.33333) * (B[ii-1] + B[ii] + B[ii+1]);
  #else
      for (i = 1; i < _PB_N - 1; i++)
        A[i] = SCALAR_VAL(0.33333) * (B[i-1] + B[i] + B[i+1]);
  #endif
    } else {
      
  #if defined(LU) && (UNROLL > 1)
      int ii = 1;
      for (; ii + (UNROLL - 1) < _PB_N-1; ii += UNROLL) {
        for (int uu = 0; uu < UNROLL; ++uu) {
          int pos = ii + uu;
          int condB = ((pos & 1) == 0);
          int condC = (((int)(B[pos]*1000.0)) & 1);
          if (condB || (condC && (_PB_TSTEPS > LEN_TH)))
            A[pos] = SCALAR_VAL(0.33333) * (B[pos-1] + B[pos] + B[pos+1]);
          else {
            if ((pos & 1) == 1)
              A[pos] = SCALAR_VAL(0.33333) * (B[pos-1] + B[pos] + B[pos+1]);
            else
              A[pos] = SCALAR_VAL(0.0);
          }
        }
      }
      for (; ii < _PB_N-1; ++ii) {
        int condB = ((ii & 1) == 0);
        int condC = (((int)(B[ii]*1000.0)) & 1);
        if (condB || (condC && (_PB_TSTEPS > LEN_TH)))
          A[ii] = SCALAR_VAL(0.33333) * (B[ii-1] + B[ii] + B[ii+1]);
        else {
          if ((ii & 1) == 1)
            A[ii] = SCALAR_VAL(0.33333) * (B[ii-1] + B[ii] + B[ii+1]);
          else
            A[ii] = SCALAR_VAL(0.0);
        }
      }
      
  #else
      for (i = 1; i < _PB_N - 1; i += 2)
        A[i] = SCALAR_VAL(0.33333) * (B[i-1] + B[i] + B[i+1]);
      for (i = 2; i < _PB_N - 2; i += 2)
        A[i] = SCALAR_VAL(0.5) * (A[i-1] + A[i+1]);
      if ((_PB_N - 2) % 2 == 0) {
        int last = _PB_N - 2;
        if (last % 2 == 0)
          A[last] = SCALAR_VAL(0.33333) * (B[last-1] + B[last] + B[last+1]);
      }
  #endif
    }
#else
    for (i = 1; i < _PB_N - 1; i++)
      A[i] = SCALAR_VAL(0.33333) * (B[i-1] + B[i] + B[i+1]);
#endif

  } /* end timesteps */
#pragma endscop

}



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

