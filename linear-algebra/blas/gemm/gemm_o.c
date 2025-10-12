#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"

/* Default tunables */
#ifndef SAMPLE_RATE
#define SAMPLE_RATE 8
#endif
#ifndef EPS
#define EPS 1e-12
#endif
#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

/* Simple deterministic xorshift32 for lightweight sampling decisions */
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
        DATA_TYPE *alpha,
        DATA_TYPE *beta,
        DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
        DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
        DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
         DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
    if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* helper inline abs for DATA_TYPE (avoids lib call) */
static inline DATA_TYPE abs_dt(DATA_TYPE x) { return x < (DATA_TYPE)0 ? -x : x; }

/* detect pow2 SAMPLE_RATE at compile time if SAMPLE_RATE is a compile-time constant */
#if defined(SAMPLE_RATE) && ( (SAMPLE_RATE & (SAMPLE_RATE - 1)) == 0 )
  #define SAMPLE_POW2 1
  #define SAMPLE_MASK (SAMPLE_RATE - 1)
#else
  #define SAMPLE_POW2 0
#endif

static
void kernel_gemm(int ni, int nj, int nk,
         DATA_TYPE alpha,
         DATA_TYPE beta,
         DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
         DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
         DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

#ifdef ENABLE_BHOT_PRECOMPUTE
  /* Precompute B-hot mask once: Bhot[k*nj + j] == 1 if |B[k][j]| > EPS */
  unsigned char *Bhot = (unsigned char*) malloc((size_t) _PB_NK * (size_t) _PB_NJ);
  if (Bhot != NULL) {
    for (k = 0; k < _PB_NK; ++k) {
      const DATA_TYPE *brow = B[k];
      size_t base = (size_t)k * (size_t)_PB_NJ;
      for (j = 0; j < _PB_NJ; ++j)
        Bhot[base + (size_t)j] = (abs_dt(brow[j]) > (DATA_TYPE)EPS) ? 1u : 0u;
    }
  } else {
    /* allocation failed: disable precompute quietly */
    Bhot = NULL;
  }
#endif

#pragma scop
  for (i = 0; i < _PB_NI; i++) {
    /* pointers to row i to help alias analysis / vectorization */
    DATA_TYPE * restrict c_row = C[i];
    const DATA_TYPE * restrict a_row = A[i];

    /* scale C row by beta if needed (special-case beta==1 skip) */
    if (! (abs_dt(beta - (DATA_TYPE)1.0) <= (DATA_TYPE)EPS) ) {
      for (j = 0; j < _PB_NJ; ++j)
        c_row[j] *= beta;
    }

#ifdef ENABLE_OPENMP
    /* If user enabled OpenMP, they can add a parallel region around outer loop.
       Here we keep deterministic per-i RNG by seeding from i. */
#endif

    /* deterministic RNG seed per i (reproducible) */
    uint32_t rng_state = (uint32_t)( (uint32_t)i * 2654435761u + 12345u );

    /* main k loop */
#ifdef ENABLE_BLOCKING
    /* optional blocking (tunable). Not enabled by default. */
    const int BK = 32; /* choose via macro if desired */
    for (int kk = 0; kk < _PB_NK; kk += BK) {
      int kmax = (kk + BK < _PB_NK) ? (kk + BK) : _PB_NK;
      for (k = kk; k < kmax; ++k) {
#else
      for (k = 0; k < _PB_NK; ++k) {
#endif
      DATA_TYPE a_ik = a_row[k];
      DATA_TYPE abs_aik = abs_dt(a_ik);
      int a_hot = (abs_aik > (DATA_TYPE)EPS);
      /* hoist alpha * a_ik (saves per-j multiply) */
      DATA_TYPE alpha_aik = alpha * a_ik;

      const DATA_TYPE * restrict b_row = B[k];

      /* UNROLL-friendly inner loop: iterate in chunks of UNROLL_FACTOR when enabled */
#ifdef ENABLE_UNROLL
      int jh;
      for (jh = 0; jh + UNROLL_FACTOR <= _PB_NJ; jh += UNROLL_FACTOR) {
        /* small unrolled chunk */
        #pragma GCC ivdep
        for (int uj = 0; uj < UNROLL_FACTOR; ++uj) {
          j = jh + uj;
#ifdef ENABLE_SAMPLING
          /* fast path: if a is hot and (optionally) Bhot indicates hot, do exact */
          if (a_hot) {
#ifdef ENABLE_BHOT_PRECOMPUTE
            if (Bhot && Bhot[(size_t)k * (size_t)_PB_NJ + (size_t)j]) {
              c_row[j] += alpha_aik * b_row[j];
              continue;
            }
#else
            DATA_TYPE bkj_tmp = b_row[j];
            if (abs_dt(bkj_tmp) > (DATA_TYPE)EPS) {
              c_row[j] += alpha_aik * bkj_tmp;
              continue;
            }
#endif
            /* else fall-through to sampling branch */
          }
          /* cold-case or sampling: draw RNG and maybe add scaled sample */
          {
            uint32_t r = xorshift32(&rng_state);
#if SAMPLE_POW2
            if ((r & SAMPLE_MASK) == 0) {
              c_row[j] += alpha_aik * b_row[j] * (DATA_TYPE)SAMPLE_RATE;
            }
#else
            if ((r % SAMPLE_RATE) == 0) {
              c_row[j] += alpha_aik * b_row[j] * (DATA_TYPE)SAMPLE_RATE;
            }
#endif
          }
#else  /* not ENABLE_SAMPLING */
          /* sampling disabled -> only update when both a and b are hot */
#ifdef ENABLE_BHOT_PRECOMPUTE
          if (a_hot && Bhot && Bhot[(size_t)k * (size_t)_PB_NJ + (size_t)j]) {
            c_row[j] += alpha_aik * b_row[j];
          }
#else
          DATA_TYPE bkj_tmp = b_row[j];
          if (a_hot && (abs_dt(bkj_tmp) > (DATA_TYPE)EPS)) {
            c_row[j] += alpha_aik * bkj_tmp;
          }
#endif
#endif /* ENABLE_SAMPLING */
        }
      }
      /* leftover j's */
      for (j = ( ( _PB_NJ / UNROLL_FACTOR) * UNROLL_FACTOR ); j < _PB_NJ; ++j) {
#ifdef ENABLE_SAMPLING
        if (a_hot) {
#ifdef ENABLE_BHOT_PRECOMPUTE
          if (Bhot && Bhot[(size_t)k * (size_t)_PB_NJ + (size_t)j]) {
            c_row[j] += alpha_aik * b_row[j];
            continue;
          }
#else
          DATA_TYPE bkj_tmp = b_row[j];
          if (abs_dt(bkj_tmp) > (DATA_TYPE)EPS) {
            c_row[j] += alpha_aik * bkj_tmp;
            continue;
          }
#endif
        }
        {
          uint32_t r = xorshift32(&rng_state);
#if SAMPLE_POW2
          if ((r & SAMPLE_MASK) == 0) {
            c_row[j] += alpha_aik * b_row[j] * (DATA_TYPE)SAMPLE_RATE;
          }
#else
          if ((r % SAMPLE_RATE) == 0) {
            c_row[j] += alpha_aik * b_row[j] * (DATA_TYPE)SAMPLE_RATE;
          }
#endif
        }
#else /* not ENABLE_SAMPLING */
#ifdef ENABLE_BHOT_PRECOMPUTE
        if (a_hot && Bhot && Bhot[(size_t)k * (size_t)_PB_NJ + (size_t)j]) {
          c_row[j] += alpha_aik * b_row[j];
        }
#else
        DATA_TYPE bkj_tmp = b_row[j];
        if (a_hot && (abs_dt(bkj_tmp) > (DATA_TYPE)EPS)) {
          c_row[j] += alpha_aik * bkj_tmp;
        }
#endif
#endif
      }
#else /* not ENABLE_UNROLL */
      /* single j loop */
      #pragma GCC ivdep
      for (j = 0; j < _PB_NJ; ++j) {
#ifdef ENABLE_SAMPLING
        if (a_hot) {
#ifdef ENABLE_BHOT_PRECOMPUTE
          if (Bhot && Bhot[(size_t)k * (size_t)_PB_NJ + (size_t)j]) {
            c_row[j] += alpha_aik * b_row[j];
            continue;
          }
#else
          DATA_TYPE bkj_tmp = b_row[j];
          if (abs_dt(bkj_tmp) > (DATA_TYPE)EPS) {
            c_row[j] += alpha_aik * bkj_tmp;
            continue;
          }
#endif
        }
        {
          uint32_t r = xorshift32(&rng_state);
#if SAMPLE_POW2
          if ((r & SAMPLE_MASK) == 0)
            c_row[j] += alpha_aik * b_row[j] * (DATA_TYPE)SAMPLE_RATE;
#else
          if ((r % SAMPLE_RATE) == 0)
            c_row[j] += alpha_aik * b_row[j] * (DATA_TYPE)SAMPLE_RATE;
#endif
        }
#else /* not ENABLE_SAMPLING */
#ifdef ENABLE_BHOT_PRECOMPUTE
        if (a_hot && Bhot && Bhot[(size_t)k * (size_t)_PB_NJ + (size_t)j])
          c_row[j] += alpha_aik * b_row[j];
#else
        DATA_TYPE bkj_tmp = b_row[j];
        if (a_hot && (abs_dt(bkj_tmp) > (DATA_TYPE)EPS))
          c_row[j] += alpha_aik * bkj_tmp;
#endif
#endif
      }
#endif /* ENABLE_UNROLL */

    } /* end k */
#ifdef ENABLE_BLOCKING
    } /* end block k loop */
#endif
  } /* end i */
#pragma endscop

#ifdef ENABLE_BHOT_PRECOMPUTE
  if (Bhot) free(Bhot);
#endif
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
          POLYBENCH_ARRAY(C),
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
           alpha, beta,
           POLYBENCH_ARRAY(C),
           POLYBENCH_ARRAY(A),
           POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

