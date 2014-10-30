/**
 *  \brief bitmerge-cuda.h
 *  \desc CUDA-based bitonic merge
 */

#if !defined (INC_BITMERGE_CUDA_H)
#define INC_BITMERGE_CUDA_H /*!< bitmerge-cuda.h included */

#include "bitmerge.h"

/** Allocate keys on the GPU, i.e., a wrapper around cudaMalloc. */
keytype* createKeysOnGPU (unsigned int n);

keytype * loadKeysOnGPU (unsigned int n);

/** Free memory previously allocated for keys on the GPU. */
void freeKeysOnGPU (keytype* A);

/** Transfers keys to the GPU, i.e., a wrapper around cudaMemcpy. */
void copyKeysToGPU (unsigned int n,
		       keytype* Dest_gpu, const keytype* Src_cpu);

/** Copy keys from the GPU, i.e., a wrapper around cudaMemcpy. */
void copyKeysFromGPU (unsigned int n,
		      keytype* Dest_cpu, const keytype* Src_gpu);

/**
 *  Performs a CUDA-based bitonic merge on a bitonic sequence,
 *  A[0:n-1]. The array A resides on the CPU, and the merged sequence
 *  will overwrite it on exit.
 */
void bitonicMerge__cuda_add_divide (unsigned int n, keytype* A_cpu);
void bitonicMerge__cuda (unsigned int n, keytype* A_cpu, int restart);

#endif

/* eof */
