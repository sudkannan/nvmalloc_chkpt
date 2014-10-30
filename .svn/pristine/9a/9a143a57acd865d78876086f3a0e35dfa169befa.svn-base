/**
 *  \brief bitmerge-cuda.cu
 *  \desc Implements a CUDA-based bitonic merge
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "bitmerge-cuda.h"
#include "timer.h"

/** Wrapper around check_gpu__, with source filename / line number info */
#define CHECK_GPU(msg)  check_gpu__ (__FILE__, __LINE__, (msg))

/**
 *  Queries the GPU to see if any errors have occured, and if so,
 *  aborts with message 'msg'.
 */
static void check_gpu__ (const char* file, size_t line, const char* msg);

static int MAX_THREADS_PER_BLOCK;
static int MAX_GRID_DIM_X;
static int MAX_GRID_DIM_Y;
static int MAX_GRID_DIM_Z;
static int MAX_SHM;

#define CU_SAFE_CALL_NO_SYNC( call ) \
{ \
	CUresult err = call; \
	if( CUDA_SUCCESS != err) { \
		fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n", \
				err, __FILE__, __LINE__ ); \
		exit(EXIT_FAILURE); \
	} \
}

static void init_param()
{
  static bool init = false;
  if (init)
    return;
  else
    init = true;

	CUresult err = cuInit(0);

	int deviceCount;
	CU_SAFE_CALL_NO_SYNC(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0)
	{
		fprintf(stderr, "There is no device supporting CUDA\n");
		return;
	}

  fprintf(stderr, "CUDA devices:");
  CUdevice dev;
	for (dev = 0; dev < deviceCount; ++dev)
  {
    char deviceName[256];
    CU_SAFE_CALL_NO_SYNC( cuDeviceGetName(deviceName, 256, dev) );
    fprintf(stderr, " \"%s\"", deviceName);
  }
  fprintf(stderr, "\n");

  // query only the first device. using multiple device would be cool though.
  dev = 0;
  CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev) );
  CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev) );
  CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev) );
  CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &MAX_GRID_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev) );
  CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &MAX_SHM, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev) );
}


__device__ void minmax_cuda(keytype* a, keytype* b)
{
	keytype a_val = *a;
	keytype b_val = *b;
	if (a_val > b_val)
	{
		*a = b_val;
		*b = a_val;
	}
}


__global__
void bitonicSplit_add_divide (unsigned int n, keytype* A, unsigned int offset)
{
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

  /* ============================================================
   *
   * Part 1 (in-class): To see that you understand how bitonic merging
   * and splitting work, fill in this function. It does not have to be
   * fast, just correct.
   *
   * Part 2 (out-of-class): Optimize your code. It's possible you will
   * want to modify the calling code also.
   *
   * ============================================================
   */

// when n=16 
// 
// 0th round:
//  offset = n/2 = 8
//  follow MSB
// 
// 1st round:
//  offset = 4
//  index: if 2nd MSB is 0, then i
//                       1,      i + offset
// 
//   0 1 2 3 8 9 10 11
//   0 1 2 3 4 5  6  7 
// + 0 0 0 0 4 4  4  4
// 
//   i + 2^2 * (i / 2^2)
// 
// 2nd round:
//  offset = 2
//  index:
//   0 1 4 5 8 9 12 13 
//   0 1 2 3 4 5  6  7
// + 0 0 2 2 4 4  6  6
// 
//   i + 2^1 * (i / 2^1)
// 
// 3nd round:
//  offset = 1
//  index:
//   0 2 4 6 8 10 12 14
//   0 1 2 3 4  5  6  7
// + 0 1 2 3 4  5  6  7
// 
//   i + 2^0 * (i / 2^0)
// 
// i + 4 * (i / 4)

  unsigned int i = gid + offset * (gid / offset);
  minmax_cuda(A + i, A + i + offset);
}


__global__
void bitonicSplit (unsigned int n, keytype* A, unsigned int offset, unsigned int shift)
{
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

  /* ============================================================
   *
   * Part 1 (in-class): To see that you understand how bitonic merging
   * and splitting work, fill in this function. It does not have to be
   * fast, just correct.
   *
   * Part 2 (out-of-class): Optimize your code. It's possible you will
   * want to modify the calling code also.
   *
   * ============================================================
   */

  unsigned int i = gid + ((gid >> shift) << shift);
  minmax_cuda(A + i, A + i + offset);
}


__global__
void bitonicSplit_9 (unsigned int n, keytype* A, unsigned int offset, unsigned int shift)
{
  const unsigned int gid = (blockIdx.x << 9) + threadIdx.x;

  /* ============================================================
   *
   * Part 1 (in-class): To see that you understand how bitonic merging
   * and splitting work, fill in this function. It does not have to be
   * fast, just correct.
   *
   * Part 2 (out-of-class): Optimize your code. It's possible you will
   * want to modify the calling code also.
   *
   * ============================================================
   */

  unsigned int i = gid + ((gid >> shift) << shift);
  minmax_cuda(A + i, A + i + offset);
}


__global__
void bitonicSplit_10 (unsigned int n, keytype* A, unsigned int offset, unsigned int shift)
{
  const unsigned int gid = (blockIdx.x << 10) + threadIdx.x;

  /* ============================================================
   *
   * Part 1 (in-class): To see that you understand how bitonic merging
   * and splitting work, fill in this function. It does not have to be
   * fast, just correct.
   *
   * Part 2 (out-of-class): Optimize your code. It's possible you will
   * want to modify the calling code also.
   *
   * ============================================================
   */

  unsigned int i = gid + ((gid >> shift) << shift);
  minmax_cuda(A + i, A + i + offset);
}


__global__
void rotate (unsigned int n, keytype* dest, keytype* src)
{
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if (gid >= n)
    return;

  dest[threadIdx.x * (n / blockDim.x) + blockIdx.x] = src[gid];
}


__global__
void rotate_22_1024 (keytype* dest, keytype* src)
{
  const unsigned int n = 4194304;
  //const unsigned int block_dim_x = 1024;
  const unsigned int gid = (blockIdx.x << 10) + threadIdx.x;
  if (gid >= n)
    return;

  dest[threadIdx.x * (n >> 10) + blockIdx.x] = src[gid];
}


__global__
void rotate_back (unsigned int n, keytype* dest, keytype* src)
{
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if (gid >= n)
    return;

  dest[(gid % (n / blockDim.x)) * blockDim.x + gid / (n / blockDim.x)] = src[gid];
}


__global__
void rotate_back_22_1024 (keytype* dest, keytype* src)
{
  const unsigned int n = 4194304;
  //const unsigned int block_dim_x = 1024;
  const unsigned int gid = (blockIdx.x << 10) + threadIdx.x;
  if (gid >= n)
    return;

  dest[((gid % (n >> 10)) << 10) + gid / (n >> 10)] = src[gid];
}


__global__
void copy_cuda (unsigned int n, keytype* dest, keytype* src)
{
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if (gid >= n)
    return;

  dest[gid] = src[gid];
}


#define __USE_CACHE
#define __DYNAMIC_SHM_ALLOC

__global__
void bitonicSplit_inblock (unsigned int n, keytype* A, unsigned int offset, unsigned int shift)
{
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if (gid >= (n >> 1))
    return;

#ifdef __USE_CACHE

#ifdef __DYNAMIC_SHM_ALLOC
  extern __shared__ keytype cache[];
#else
  __shared__ keytype cache[1024 * 2];
#endif

  unsigned int i = gid + ((gid >> shift) << shift);

  cache[threadIdx.x] = A[i];
  cache[threadIdx.x + offset] = A[i + offset];

  unsigned int offset_ = offset;

  __syncthreads();

  while (true)
  {
    unsigned int i = threadIdx.x + ((threadIdx.x >> shift) << shift);
    minmax_cuda(cache + i, cache + i + offset_);

    offset_ >>= 1;
    -- shift;

    __syncthreads();

    // It has to be here. A last sync is needed, since at the last round each
    // thread is working on the offset other than the offset that copies back
    // from cache to the original A.
    if (offset_ == 0)
      break;
  }

  A[i] = cache[threadIdx.x];
  A[i + offset] = cache[threadIdx.x + offset];

#else // __USE_CACHE
  while (true)
  {
    unsigned int i = gid + ((gid >> shift) << shift);
    minmax_cuda(A + i, A + i + offset);

    offset >>= 1;
    -- shift;

    if (offset == 0)
      break;

    __syncthreads();
  }
#endif  // __USE_CACHE
}


__global__
void bitonicSplit_inblock_9 (unsigned int n, keytype* A)
{
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if (gid >= (n >> 1))
    return;

  const unsigned int shift = 9;
  const unsigned int offset = 512;

  __shared__ keytype cache[512 * 2];

  unsigned int i = gid + ((gid >> shift) << shift);

  cache[threadIdx.x] = A[i];
  cache[threadIdx.x + offset] = A[i + offset];

  __syncthreads();

  unsigned int j;
  j = threadIdx.x + ((threadIdx.x >>  9) <<  9); minmax_cuda(cache + j, cache + j +  512); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  8) <<  8); minmax_cuda(cache + j, cache + j +  256); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  7) <<  7); minmax_cuda(cache + j, cache + j +  128); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  6) <<  6); minmax_cuda(cache + j, cache + j +   64); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  5) <<  5); minmax_cuda(cache + j, cache + j +   32); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  4) <<  4); minmax_cuda(cache + j, cache + j +   16); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  3) <<  3); minmax_cuda(cache + j, cache + j +    8); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  2) <<  2); minmax_cuda(cache + j, cache + j +    4); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  1) <<  1); minmax_cuda(cache + j, cache + j +    2); __syncthreads();
  j = threadIdx.x + threadIdx.x;                 minmax_cuda(cache + j, cache + j +    1); __syncthreads();

  A[i] = cache[threadIdx.x];
  A[i + offset] = cache[threadIdx.x + offset];
}


__global__
void bitonicSplit_inblock_10 (unsigned int n, keytype* A)
{
  const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if (gid >= (n >> 1))
    return;

  const unsigned int shift = 10;
  const unsigned int offset = 1024;

  __shared__ keytype cache[1024 * 2];

  unsigned int i = gid + ((gid >> shift) << shift);

  cache[threadIdx.x] = A[i];
  cache[threadIdx.x + offset] = A[i + offset];

  __syncthreads();

  unsigned int j;
  j = threadIdx.x + ((threadIdx.x >> 10) << 10); minmax_cuda(cache + j, cache + j + 1024); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  9) <<  9); minmax_cuda(cache + j, cache + j +  512); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  8) <<  8); minmax_cuda(cache + j, cache + j +  256); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  7) <<  7); minmax_cuda(cache + j, cache + j +  128); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  6) <<  6); minmax_cuda(cache + j, cache + j +   64); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  5) <<  5); minmax_cuda(cache + j, cache + j +   32); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  4) <<  4); minmax_cuda(cache + j, cache + j +   16); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  3) <<  3); minmax_cuda(cache + j, cache + j +    8); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  2) <<  2); minmax_cuda(cache + j, cache + j +    4); __syncthreads();
  j = threadIdx.x + ((threadIdx.x >>  1) <<  1); minmax_cuda(cache + j, cache + j +    2); __syncthreads();
  j = threadIdx.x + threadIdx.x;                 minmax_cuda(cache + j, cache + j +    1); __syncthreads();

  A[i] = cache[threadIdx.x];
  A[i + offset] = cache[threadIdx.x + offset];
}


void bitonicMerge__cuda_add_divide (unsigned int n, keytype* A)
{
  init_param();

  /* Copy A to A_gpu, which resides on the GPU */
  keytype* A_gpu = createKeysOnGPU (n);
  copyKeysToGPU (n, A_gpu, A);

  /* Start timer, _after_ CPU-GPU copies */
  stopwatch_t* timer = stopwatch_create (); assert (timer);
  stopwatch_start (timer);

  /* ============================================================
   *
   * Part 1 (in-class): Make sure you understand what the following
   * while loop does.
   *
   * Part 2 (out-of-class): Optimize this code. You may, if you wish,
   * want to modify the bitonic split kernel.
   *
   * ============================================================
   */
  const unsigned int n_half = n >> 1; /* n/2 */
  const unsigned int BLOCKSIZE = MAX_THREADS_PER_BLOCK;
  const unsigned int NUM_BLOCKS = (n_half + BLOCKSIZE - 1) / BLOCKSIZE;
  assert (isPower2 (n) && isPower2 (BLOCKSIZE));

  unsigned int offset = n_half;
  while (offset) {
    bitonicSplit_add_divide <<<NUM_BLOCKS, BLOCKSIZE>>> (n, A_gpu, offset);
    offset >>= 1;
  }
  cudaDeviceSynchronize ();

  /* Stop timer and report bandwidth _without_ the CPU-GPU copies */
  long double t_merge_nocopy = stopwatch_stop (timer);
  long double bw_merge_nocopy = estimateBitonicBandwidth (n, t_merge_nocopy);
  printf ("==> (CUDA) %u blocks of size %u each:"
	  " %Lg secs (~ %.2Lf GB/s)\n",
	  NUM_BLOCKS, BLOCKSIZE, t_merge_nocopy, bw_merge_nocopy);

  /* Copy results back to CPU and return */
  copyKeysFromGPU (n, A, A_gpu);
  freeKeysOnGPU (A_gpu);
}


void bitonicMerge__cuda (unsigned int n, keytype* A)
{
  init_param();

  /* Copy A to A_gpu, which resides on the GPU */
  keytype* A_gpu = createKeysOnGPU (n);
  copyKeysToGPU (n, A_gpu, A);

//  keytype* tmp_gpu = createKeysOnGPU (n);

  /* Start timer, _after_ CPU-GPU copies */
  stopwatch_t* timer = stopwatch_create (); assert (timer);
  stopwatch_start (timer);

  /* ============================================================
   *
   * Part 1 (in-class): Make sure you understand what the following
   * while loop does.
   *
   * Part 2 (out-of-class): Optimize this code. You may, if you wish,
   * want to modify the bitonic split kernel.
   *
   * ============================================================
   */
  const unsigned int n_half = n >> 1; /* n/2 */
  //const unsigned int BLOCKSIZE = MAX_THREADS_PER_BLOCK;
  const unsigned int BLOCKSIZE = 512;
  const unsigned int NUM_BLOCKS = (n_half + BLOCKSIZE - 1) / BLOCKSIZE;
  assert (isPower2 (n) && isPower2 (BLOCKSIZE));

  unsigned int offset = n_half;
  unsigned int shift = 0;
  for (unsigned int t_ = 1; t_ != n_half; t_ <<= 1)
	  ++ shift;

//  {
//    //while (offset > BLOCKSIZE) {
//    while (shift > 20) {
//      bitonicSplit <<<NUM_BLOCKS, BLOCKSIZE>>> (n, A_gpu, offset, shift);
//      offset >>= 1;
//      -- shift;
//    }
//
//    unsigned int num_transpose_blocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
//
//    // rotate
//    bool rotated = false;
//    if (shift == 20)
//    {
//      rotated = true;
//      copy_cuda <<<num_transpose_blocks, BLOCKSIZE>>> (n, tmp_gpu, A_gpu);  // 0.000605 sec
//      //rotate <<<num_transpose_blocks, BLOCKSIZE>>> (n, A_gpu, tmp_gpu);
//      rotate_22_1024 <<<num_transpose_blocks, BLOCKSIZE>>> (A_gpu, tmp_gpu);     // saves 0.000031 sec
//    }
//
//    if (shift > 10)
//    {
//      shift = 10;
//      offset = 1024;
//    }
//    bitonicSplit_inblock_10 <<<NUM_BLOCKS, BLOCKSIZE>>> (n, A_gpu);
//
//    // rotate back
//    if (rotated)
//    {
//      copy_cuda <<<num_transpose_blocks, BLOCKSIZE>>> (n, tmp_gpu, A_gpu);
//      //rotate_back <<<num_transpose_blocks, BLOCKSIZE>>> (n, A_gpu, tmp_gpu);  // 0.003686 sec
//      rotate_back_22_1024 <<<num_transpose_blocks, BLOCKSIZE>>> (A_gpu, tmp_gpu);  // saves 0.000081 sec
//    }
//
//    // total: 0.007819 sec
//
//    if (shift > 10)
//    {
//      shift = 10;
//      offset = 1024;
//    }
//    bitonicSplit_inblock_10 <<<NUM_BLOCKS, BLOCKSIZE>>> (n, A_gpu);
//  }

  {
    if (BLOCKSIZE == 512)
    {
      while (offset > BLOCKSIZE) {
        bitonicSplit_9 <<<NUM_BLOCKS, BLOCKSIZE>>> (n, A_gpu, offset, shift);
        offset >>= 1;
        -- shift;
      }
    }
    else
    {
      while (offset > BLOCKSIZE) {
        bitonicSplit <<<NUM_BLOCKS, BLOCKSIZE>>> (n, A_gpu, offset, shift);
        offset >>= 1;
        -- shift;
      }
    }

    if (shift == 9)
      bitonicSplit_inblock_9 <<<NUM_BLOCKS, BLOCKSIZE>>> (n, A_gpu); // 0.000965 sec
    else
      bitonicSplit_inblock <<<NUM_BLOCKS, BLOCKSIZE, 2 * BLOCKSIZE * sizeof(keytype)>>> (n, A_gpu, offset, shift);
  }

  cudaDeviceSynchronize ();

  /* Stop timer and report bandwidth _without_ the CPU-GPU copies */
  long double t_merge_nocopy = stopwatch_stop (timer);
  long double bw_merge_nocopy = estimateBitonicBandwidth (n, t_merge_nocopy);
  printf ("==> (CUDA) %u blocks of size %u each:"
	  " %Lg secs (~ %.2Lf GB/s)\n",
	  NUM_BLOCKS, BLOCKSIZE, t_merge_nocopy, bw_merge_nocopy);

  /* Copy results back to CPU and return */
  copyKeysFromGPU (n, A, A_gpu);
  freeKeysOnGPU (A_gpu);
//  freeKeysOnGPU (tmp_gpu);
}

static void check_gpu__ (const char* file, size_t line, const char* msg)
{
  cudaError_t err = cudaGetLastError ();
  if (err != cudaSuccess) {
    fprintf (stderr, "*** [%s:%lu] %s -- CUDA Error (%d): %s ***\n",
	     file, line, msg, (int)err, cudaGetErrorString (err));
    exit (-1);
  }
}

keytype *
createKeysOnGPU (unsigned int n)
{
  keytype* A_gpu = NULL;
  if (n) {
    cudaMalloc (&A_gpu, n * sizeof (keytype)); CHECK_GPU ("Out of memory?");
    assert (A_gpu);
  }
  return A_gpu;
}

void
freeKeysOnGPU (keytype* A_gpu)
{
  if (A_gpu) cudaFree (A_gpu);
}

void
copyKeysToGPU (unsigned int n, keytype* Dest_gpu, const keytype* Src_cpu)
{
  cudaMemcpy (Dest_gpu, Src_cpu, n * sizeof (keytype),
	      cudaMemcpyHostToDevice);  CHECK_GPU ("Copying keys to GPU");
}

void
copyKeysFromGPU (unsigned int n, keytype* Dest_cpu, const keytype* Src_gpu)
{
  cudaMemcpy (Dest_cpu, Src_gpu, n * sizeof (keytype),
	      cudaMemcpyDeviceToHost);  CHECK_GPU ("Copying keys from GPU");
}

/* eof */
