// -*- mode:c++; tab-width:2; indent-tabs-mode:nil;  -*-
/**
 *  \file bitmerge.cc
 *
 *  \brief Lab 3: Bitonic merge driver program, set up for evaluating
 *  sequential, Cilk Plus, and CUDA implementations.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer.h"
#include "bitmerge.h"
#include "bitmerge-seq.h"
#include "bitmerge-cilk.h"
#include "bitmerge-cuda.h"

/** Returns 1 if 'n' is "small enough */
#define DEBUG(n)  ((n) <= 16)

/* ============================================================
 */
int 
main (int argc, char** argv)
{
  if (argc < 2) {
    fprintf (stderr, "usage: %s <n>\n", argv[0]);
    return -1;
  }

  int n = atoi (argv[1]);
  printf ("n: %d\n", n);

  assert(argv[2]);
  int restart = atoi(argv[2]);


  assert (isPower2 (n));
	
  /* Prepare timer */
  struct stopwatch_t* timer = stopwatch_create (); assert (timer);
  stopwatch_init ();

  /* Create a bitonic input sequence */
  keytype* input = (keytype*) malloc (n * sizeof (keytype));
  assert (input);
  bzero (input, n * sizeof (keytype));
  srand48 (1); /* For varying seeds, may change '1' to, say, 'time (NULL)' */
  createBitonic (n, n, input);

  /* For debugging purposes, print key values when n is small */
  debugPrintKeys (DEBUG(n), "Input ", n, input);
  assert (isBitonic (n, input));

  /* Call a trusted sequential CPU code to compute the desired answer */
  keytype* solution = (keytype*) malloc (n * sizeof (keytype));
  assert (solution);
  copyKeys (n, solution, input);
  stopwatch_start (timer);
  qsort (solution, n, sizeof (keytype), compareKeysAscending);
  long double t_qsort = stopwatch_stop (timer); /* seconds */
  long double bw_qsort = estimateBitonicBandwidth (n, t_qsort);
  fprintf (stderr, "Quicksort: %Lg secs (~ %.2Lf GB/s)\n", t_qsort, bw_qsort);

  debugPrintKeys (DEBUG(n), "Solution ", n, solution);

  /* Benchmark different implementations. For each implementation, we
   * start with an untimed "warm-up" run to check the answer, and then
   * do a "real" timed run.
   */
  keytype* computed = (keytype *)malloc (n * sizeof (keytype));

  {
    fprintf (stderr, "... sequential bitonic merge ...\n");
    copyKeys (n, computed, input);
    bitonicMerge__seq (n, computed);
    assertEqualKeys (n, solution, computed);
    debugPrintKeys (DEBUG(n), "Sequential ", n, computed);

    copyKeys (n, computed, input);
    stopwatch_start (timer);
    bitonicMerge__seq (n, computed);
    long double t_seq = stopwatch_stop (timer); /* seconds */
    long double bw_seq = estimateBitonicBandwidth (n, t_seq);
    printf ("Sequential: %Lg secs (~ %.2Lf GB/s)\n", t_seq, bw_seq);
  }

  {
#if 0
    /*fprintf (stderr, "... Cilk Plus-based bitonic merge ...\n");
    copyKeys (n, computed, input);
    bitonicMerge__cilk (n, computed);
    assertEqualKeys (n, solution, computed);
    debugPrintKeys (DEBUG(n), "Cilk Plus ", n, computed);

    copyKeys (n, computed, input);
    stopwatch_start (timer);
    bitonicMerge__cilk (n, computed);
    long double t_cilk = stopwatch_stop (timer);  seconds */
    long double bw_cilk = estimateBitonicBandwidth (n, t_cilk);
    printf ("Cilk Plus: %Lg secs (~ %.2Lf GB/s)\n", t_cilk, bw_cilk);
#endif
  }

//  {
//    fprintf (stderr, "... CUDA-based bitonic merge (using add/divide) ...\n");
//    copyKeys (n, computed, input);
//    bitonicMerge__cuda_add_divide (n, computed);
//    assertEqualKeys (n, solution, computed);
//    debugPrintKeys (DEBUG(n), "CUDA ", n, computed);
//
//    copyKeys (n, computed, input);
//    stopwatch_start (timer);
//    bitonicMerge__cuda_add_divide (n, computed);
//    long double t_cuda = stopwatch_stop (timer); /* seconds */
//    long double bw_cuda = estimateBitonicBandwidth (n, t_cuda);
//    printf ("CUDA (including PCIe copies): %Lg secs (~ %.2Lf GB/s)\n", t_cuda, bw_cuda);
//  }

  {
    fprintf (stderr, "... CUDA-based bitonic merge ...\n");
    /*copyKeys (n, computed, input);
    bitonicMerge__cuda(n, computed, restart);
    debugPrintKeys (DEBUG(n), "CUDA ", n, computed);
    asertEqualKeys (n, solution, computed);*/

    copyKeys (n, computed, input);
    stopwatch_start (timer);
    bitonicMerge__cuda(n, computed, restart);
    long double t_cuda = stopwatch_stop (timer); /* seconds */
    long double bw_cuda = estimateBitonicBandwidth (n, t_cuda);
    printf ("CUDA (including PCIe copies): %Lg secs (~ %.2Lf GB/s)\n", t_cuda, bw_cuda);
  }

  /* Clean-up */
  free (computed);
  free (solution);
  free (input);
  stopwatch_destroy (timer);
  return 0;
}

/* ============================================================
 * Helper functions (see interfaces in bitmerge.h)
 */

void minmax (keytype* a, keytype* b)
{
  assert (a && b);
  keytype a_val = *a, b_val = *b;
  if (a_val > b_val) {
    *a = b_val;
    *b = a_val;
  }
}

void assertEqualKeys (unsigned int n, const keytype* solution,
                      const keytype* output)
{
  for (unsigned int i = 0; i < n; ++i) {
    if (solution[i] != output[i]) {
      fprintf (stderr,
               "*** Incorrect output at position %u:"
               " observed %g rather than %g ***\n",
               i, (double)output[i], (double)solution[i]);
      assert (0);
    }
  } /* i */
}

void debugPrintKeys (int enabled, const char* tag,
                     unsigned int n, const keytype* A)
{
  if (!enabled) return;
  fprintf (stderr, "%s", tag ? tag : "");
  printKeys (stderr, n, A);
  fprintf (stderr, "\n");
}

int isPower2 (unsigned int N)
{
  if (N == 0)
    return 0;
  else {
    while ((N & 1) == 0)
      N = N >> 1;
    return N == 1;
  }
}

int isSortedAscending (unsigned int n, const keytype* A)
{
  for (unsigned int i = 1; i < n; ++i)
    if (A[i] < A[i-1]) return 0;
  return 1;
}

int
compareKeysAscending (const void* pa, const void* pb)
{
  keytype a = *((const keytype *)pa);
  keytype b = *((const keytype *)pb);
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

void copyKeys (unsigned int n, keytype* Dest, const keytype* Src)
{
  memcpy (Dest, Src, n * sizeof (keytype));
}

void printKeys (FILE* fp, unsigned int n, const keytype* A)
{
  const int KEYS_PER_LINE = 8;
  fprintf (fp, "[%u keys]", n);
  for (unsigned int i = 0; i < n; ++i) {
    if ((i % KEYS_PER_LINE) == 0)
      fprintf (fp, "\n   ");
    fprintf (fp, " %.15g", (double)A[i]);
  } /* i */
  fprintf (fp, "\n");
}

void createBitonic (unsigned int N, unsigned int K, keytype *A)
{
  assert (A || !N || !K);

  if ((N % K) != 0) {
    fprintf (stderr, "K does not divide N\n");
    exit (-1);
  } else {
    /* for each sub sequence */
    for (int i = 0; i < N; i += K) {
      /* create a bitonic sequence */
      for (int j = i; j < i + K / 2; j++) {
        if (j == i) {
          A[j] = drand48 ();
        } else {
          A[j] = A[j - 1] + drand48 ();
        }
      } /* j (1) */
      for (int j = i + K / 2; j < i + K; j++) {
        A[j] = A[j - 1] - drand48 ();
      } /* j (2) */
    } /* i */
  } /* if */
}

int isBitonic (unsigned int n, const keytype* A)
{
  assert (A || !n);
  if (n <= 1) /* trivially bitonic */
    return 1;

  assert (n >= 2);

  /* determine starting direction: 1 for non-decreasing, 0 for increasing */
  unsigned int i = 1;
  unsigned int dir = (A[i-1] <= A[i]);
  while (++i < n) {
    unsigned int dir_i = (A[i-1] <= A[i]);
    if (dir != dir_i) { /* change in directions */
      dir = dir_i;
      break;
    }
  }
  while (++i < n) {
    unsigned int dir_i = (A[i-1] < A[i]);
    if (dir != dir_i) { /* another change in directions ==> not bitonic */
      return 0;
    }
  }

  /* No more than 1 change in direction occured ==> bitonic */
  return 1;
}

long double estimateBitonicBandwidth (unsigned int n, long double t)
{
  /* This bandwidth estimate optimistically assumes two passes (read &
     write) of the data */
  return (long double)2e-9 * n * sizeof (keytype) / t;
}

/* eof */
