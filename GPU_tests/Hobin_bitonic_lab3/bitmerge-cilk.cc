/**
 *  \brief bitmerge-cilk.cc
 *  \desc Implements a Cilk Plus-based bitonic merge
 */

#include <assert.h>

#include "bitmerge-cilk.h"
#include "bitmerge-seq.h"

void bitonicSplitFwd__cilk (unsigned int n, keytype* A)
{
  assert (n == 1 || (n % 2) == 0);
  const unsigned int n_half = n >> 1;
  _Cilk_for (unsigned int i = 0; i < n_half; ++i)
    minmax (&A[i], &A[n_half+i]);
}

void bitonicMergeHelper__cilk (unsigned int n, keytype* A, unsigned int G)
{
  if (n <= G)
    bitonicMerge__seq (n, A);
  else { /* n >= G */
    assert (n >= 2);
    bitonicSplitFwd__cilk (n, A);
    const unsigned int n_half = n >> 1;
    _Cilk_spawn bitonicMergeHelper__cilk (n_half, A, G);
    bitonicMergeHelper__cilk (n_half, A + n_half, G);
  }
}

void bitonicMerge__cilk (unsigned int n, keytype* A)
{
  unsigned int G = 131072;
  bitonicMergeHelper__cilk (n, A, G);
}

/* eof */
