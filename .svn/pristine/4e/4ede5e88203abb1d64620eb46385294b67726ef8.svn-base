/**
 *  \file bitmerge.h
 *  \brief Bitonic sort helper functions.
 */

#if !defined (INC_BITMERGE_H)
#define INC_BITMERGE_H /*!< bitmerge.h included */

#include <stdio.h>

/** Type of keys to sort */
typedef float keytype;

/** qsort-compatible key-comparison function for ascending sequences */
int compareKeysAscending (const void* a, const void* b);

/** Copy keys */
void copyKeys (unsigned int n, keytype* Dest, const keytype* Src);

/** Prints elements of an array to a file */
void printKeys (FILE* fp, unsigned int n, const keytype* A);

/** Returns 1 iff A is sorted in ascending order */
int isSortedAscending (unsigned int n, const keytype* A);

/** Returns the smaller of two values */
#define MIN(x,y) ((x < y) ? x : y)

/** Returns the larger of two values */
#define MAX(x,y) ((x < y) ? y : x)

/** Returns 1 iff N is a power of 2 */
int isPower2 (unsigned int n);

/** Creates N/K bitonic sequences of length K each. */
void createBitonic (unsigned int N, unsigned int K, keytype *A);

/** Checks whether a sequence is bitonic */
int isBitonic (unsigned int n, const keytype* A);

/** Overwrites (*a, *b) with (min (*a, *b), max (*a, *b)) */
void minmax (keytype* a, keytype* b);

/** Overwrites (*a, *b) with (max (*a, *b), min (*a, *b)) */
#define maxmin(a,b)  minmax ((b), (a))

/**
 *  Given some execution time 't', this routine estimates the
 *  effective bandwidth (GB/s) of streaming through 'n' keys twice.
 */
long double estimateBitonicBandwidth (unsigned int n, long double t);

/** Returns 1 iff the array 'output' exacty matches 'solution' */
void assertEqualKeys (unsigned int n, const keytype* soln, const keytype* A);

/** Prints keys to 'stderr', when 'enabled' is non-zero */
void debugPrintKeys (int enabled, const char* tag,
                     unsigned int n, const keytype* A);

#endif

/* eof */
