/* -*- mode:c++; tab-width:2; indent-tabs-mode:nil;  -*- */
/**
 *  \file listrank.cc
 *  \brief List ranking benchmark driver.
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "timer.c"

/*Headers for reducer*/
#include <cilk/cilk.h> 
#include <cilk/cilk_api.h> 
#include <cilk/reducer_opor.h>

#define NIL -1 /*!< "NULL" pointer, marking end of a list */

/* ======================================================================
 *  Some helper routines
 */

/* Swaps two integers */
static void swapInts (int* pa, int* pb);

/* Swaps two integer pointers */
static void swapIntPointers (int** pa, int** pb);

/** Returns a newly allocated array of 'n' integers */
int* createInts (int n);

/** Release memory of a previously allocated array */
void freeInts (int* A);

/** Returns a newly allocated copy of an array of integers */
int* duplicateInts (int n, const int* A);

/** Copies values from Src to Dest */
void copyInts (int n, int* Dest, const int* Src);

/** Sets all array values to 0 */
void zeroInts (int n, int* A);

/* ====================================================================== */

/**
 *  Given a linked list represented as an array pool, this routine
 *  computes the rank, 'Rank[i]', or distance to the tail, of each
 *  node 'i'. The head of the list is at position 'head', i.e., the
 *  node represented by 'Rank[head]' and 'Next[head]'. 'head' may be
 *  'NIL' if the list is empty.
 */
void
rankList__seq (int n, int* Rank, const int* Next, int head)
{
  if (n == 0 || head == NIL) return; /* pool or list are empty */

  /* What does this loop do? */
  /* This loop calculates the rank of the head node by traversing the list */
  int cur_node = head;
  int count = 0;
  do {
    ++count;
    cur_node = Next[cur_node];
  } while (cur_node != NIL);

  /* What does this loop do? */
  /*Then calculates the rank of all other elements by decrementing
  the rank of the head node calculated in the first loop*/
  cur_node = head;
  do {
    Rank[cur_node] = --count;
    cur_node = Next[cur_node];
  } while (cur_node != NIL);
}


bool check_next(int *Next, int n)
{
    cilk::reducer_opor<bool> notnil(false);
	 _Cilk_for(int i=0; i < n; i++)
	{
		if(Next[i] != -1){
			notnil.set_value(true);
         }
	}
    bool flag = notnil.get_value(); 
	return flag;
}

/* ====================================================================== */

/**
 *  Given a linked list represented as a pool (array), computes the
 *  rank Rank[i], or distance to the tail, of each node i.
 */
void
rankList__par (int n, int* Rank, const int* Next)
{
  if (n == 0) return; // empty pool

  /* Initialize to all ones _except_ the tail, which gets a 0 */
  _Cilk_for (int i = 0; i < n; ++i)
    Rank[i] = (Next[i] == NIL) ? 0 : 1;

  /* To help implement synchronization, it may be helpful to have
     additional buffers. */
  int* Rank_cur = Rank;
  int* Next_cur = duplicateInts (n, Next); // function we've provided
  int* Rank_next = createInts (n);
  int* Next_next = duplicateInts (n, Next);

  /*------------------------------------------------------------
   *
   * ... YOUR CODE GOES HERE ...
   *
   * (you may also modify the preceding code if you wish)
   *
   * ------------------------------------------------------------
   */
#if 1 
	//while(check_next(Next_next, n)) {
for (int j = 1; j < n; j = j*2) {
	 _Cilk_for(int i = 0; i < n; ++i){
	   if (Next_cur[i] != NIL){
	     Rank_next[i] = Rank_cur[Next_cur[i]];
    	 Next_next[i] = Next_cur[Next_cur[i]];
		}
	}

	 _Cilk_for(int i = 0; i < n; ++i){
		 if (Next_cur[i] != NIL){
		Rank_cur[i] += Rank_next[i];
		Next_cur[i] = Next_next[i];

		}
	 }
	}	
#endif

  /* If you use the extra buffers, be sure to clean them up. */
  if (Rank != Rank_cur) {
    copyInts (n, Rank, Rank_cur);
    freeInts (Rank_cur);
  } else { freeInts (Rank_next); }
  freeInts (Next_next);
  freeInts (Next_cur);
}

/* ====================================================================== */

static
void
swapInts (int* pa, int* pb)
{
  assert (pa != NULL && pb != NULL);
  int t = *pa;
  *pa = *pb;
  *pb = t;
}

static
void
swapIntPointers (int** pa, int** pb)
{
  assert (pa != NULL && pb != NULL);
  int* t = *pa;
  *pa = *pb;
  *pb = t;
}

int *
createInts (int n)
{
  int* newArray = NULL;
  if (n > 0) {
    newArray = (int *)malloc (n * sizeof (int));
    assert (newArray != NULL);
  }
  return newArray;
}

void
freeInts (int* A)
{
  if (A != NULL) free (A);
}

int *
duplicateInts (int n, const int* A)
{
  int* A_copy = createInts (n);
  copyInts (n, A_copy, A);
  return A_copy;
}

void
copyInts (int n, int* Dest, const int* Src)
{
  if (n > 0)
    memcpy (Dest, Src, n * sizeof (int));
}

void
zeroInts (int n, int* A)
{
  bzero (A, n * sizeof (int));
}

/** Generates a uniform random permutation of an array */
void
shuffle (int n, int* A)
{
  /* Implements the Fisher-Yates (Knuth) algorithm:
   * http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
   */
  for (int i = 0; i < (n-1); ++i)
    swapInts (&(A[i]), &(A[i+1+(lrand48 () % (n-i-1))]));
}

/**
 *  Allocates a pool of 'next' pointers, and initializes into a single
 *  random linked list. The head is the first element ('next[0]') and
 *  the tail is the element whose next pointer is -1 (i.e., the
 *  element 'k' such that 'next[k] == NIL').
 */
int *
createRandomList (int n)
{
  /* Create an initial linked list where each node i points to i+1. */
  int* Init = createInts (n);
  for (int i = 0; i < n; ++i)
    Init[i] = i + 1;
  Init[n-1] = NIL; /* "NULL" pointer */

  /* Remap node i > 0 to position AddrMap[i]. */
  int* AddrMap = createInts (n);
  for (int i = 0; i < n; ++i)
    AddrMap[i] = i;
  shuffle (n-1, AddrMap+1);

  /* Create final list */
  int* Next = createInts (n);
  for (int i = 0; i < n; ++i)
    Next[AddrMap[i]] = Init[i] > 0 ? AddrMap[Init[i]] : NIL;

  freeInts (AddrMap);
  freeInts (Init);
  return Next;
}

/* ====================================================================== */

const static int MAX_PRINT = 16; /*!< Maximum list length to print */

void
printList (const char* tag, const int* Rank, const int* Next, int head)
{
  int cur_node;

  fprintf (stderr, "=== %s ===\n", tag ? tag : "");
  fprintf (stderr, "  Rank: [");
  cur_node = head;
  while (cur_node != NIL) {
    fprintf (stderr, " %d", Rank[cur_node]);
    cur_node = Next[cur_node];
  }
  fprintf (stderr, " ]\n");

  fprintf (stderr, "  Next: [");
  cur_node = head;
  while (cur_node != NIL) {
    fprintf (stderr, " %d", Next[cur_node]);
    cur_node = Next[cur_node];
  }
  fprintf (stderr, " ]\n");
}

void
check (int n, const int* R, const int* R_true)
{
  for (int i = 0; i < n; ++i)
    if (R[i] != R_true[i]) {
      fprintf (stderr, "*** ERROR: *** [i=%d] Rank %d != %d ***\n", i, R[i], R_true[i]);
      assert (0);
    }
  fprintf (stderr, "    (OK!)\n");
}

/**
 *  Given a list size 'n' and the execution time 't' in seconds,
 *  returns an estimate of the effective bandwidth (in bytes per
 *  second) of list ranking.
 */
long double
estimateBandwidth (int n, long double t)
{
  return (long double)n * (2*sizeof (int) + sizeof (int)) / t;
}

/* ====================================================================== */

int
main (int argc, char* argv[])
{
  if (argc != 3) {
    fprintf (stderr, "usage: %s <n> <trials>\n", argv[0]);
    return -1;
  }

  int N = atoi (argv[1]); assert (N > 0);
  int NTRIALS = atoi (argv[2]); assert (NTRIALS > 0);

  fprintf (stderr, "N: %d\n", N);
  fprintf (stderr, "Node size: %lu (index) + %lu (rank) bytes\n", sizeof (int), sizeof (int));
  fprintf (stderr, "No. of timing trials: %d\n", NTRIALS);

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create ();
  assert (timer);

  for (int trial = 0; trial < NTRIALS; ++trial) {
    fprintf (stderr, "\n... [%d] Creating node pool ...\n", trial);
    int* Next = createRandomList (N);

    fprintf (stderr, "... Running the sequential algorithm ...\n");
    int* Rank_seq = createInts (N);
    zeroInts (N, Rank_seq);
    if (N <= MAX_PRINT) printList ("Sequential: before", Rank_seq, Next, 0);
    stopwatch_start (timer);
    rankList__seq (N, Rank_seq, Next, 0 /* head */);
    long double t_seq = stopwatch_stop (timer); /* seconds */
    long double bw_seq = estimateBandwidth (N, t_seq) * 1e-9; /* GB/s */
    fprintf (stderr, "    (Done: %Lg sec, %Lg GB/s.)\n", t_seq, bw_seq);
    if (N <= MAX_PRINT) printList ("Sequential: after", Rank_seq, Next, 0);

    fprintf (stderr, "\n... Running the parallel algorithm ...\n");
    int* Rank_par = createInts (N);
    zeroInts (N, Rank_par);
    if (N <= MAX_PRINT) printList ("Parallel: before", Rank_par, Next, 0);
    stopwatch_start (timer);
    rankList__par (N, Rank_par, Next);
    long double t_par = stopwatch_stop (timer); /* seconds */
    long double bw_par = estimateBandwidth (N, t_par) * 1e-9; /* GB/s */
    fprintf (stderr, "    (Done: %Lg sec, %Lg GB/s.)\n", t_par, bw_par);
    if (N <= MAX_PRINT) printList ("Parallel: after", Rank_par, Next, 0);

    printf ("%d %d %lu %lu %Lg %Lg %Lg %Lg\n",
            trial, N, sizeof (int), sizeof (int),
            t_seq, bw_seq, t_par, bw_par);

    fprintf (stderr, "... Checking the answer ...\n");
    check (N, Rank_par, Rank_seq);

    freeInts (Rank_par);
    freeInts (Rank_seq);
    freeInts (Next);
  }

  stopwatch_destroy (timer);
  return 0;
}

/* eof */
