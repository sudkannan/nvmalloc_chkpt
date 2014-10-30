/**
 *  \file bitmerge-seq.cc
 *
 *  \brief Implements a sequential bitonic merge.
 */

#include <assert.h>
#include "bitmerge-seq.h"

void bitonicMerge__seq (unsigned int n, keytype* A)
{
  if (n <= 1) return;
  bitonicSplitFwd__seq (n, A);
  if (n > 2) {
    const unsigned int n_half = n >> 1;
    bitonicMerge__seq (n_half, A);
    bitonicMerge__seq (n_half, A + n_half);
  }
}

void bitonicSplitFwd__seq (unsigned int n, keytype* A)
{
  assert (n == 1 || (n % 2) == 0);
  const unsigned int n_half = n >> 1;
  for (unsigned int i = 0; i < n_half; ++i)
    minmax (&A[i], &A[n_half+i]);
}

/* eof */
