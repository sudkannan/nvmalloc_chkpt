#include <stdlib.h>
#include <stdio.h>

#include "transpose.h"
#include "timer.h"


__global__ 
void matTranspose_naive(float* B, float* A) {

  /* get global index for this thread */
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  /* transpose */
  B[j * N + i] = A[i * N + j];
}


#define FIX_ME 0

__global__ 
void matTranspose_sm(float* B, float* A)  {
  float val;
  __shared__ float cache[BLOCK_SIZE][BLOCK_SIZE];

  /* get global index for this thread */
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;


  /* load data into shared memory so that it is transposed */
  cache[threadIdx.y][threadIdx.x] = A[i * N + j];
  __syncthreads();

  /* fetch the right data back from the shared memory */
  val = cache[threadIdx.x][threadIdx.y];

  /* Write the data back to main memory */
  i = blockIdx.x * blockDim.y + threadIdx.y;
  j = blockIdx.y * blockDim.x + threadIdx.x;
  B[i * N + j] = val;
}

__global__ 
void matTranspose_sm1(float* B, float* A)  {
  float val;
  __shared__ float cache[BLOCK_SIZE][BLOCK_SIZE];

  /* get global index for this thread */
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;


  /* load data into shared memory so that it is transposed */
  cache[FIX_ME][FIX_ME] = A[i * N + j];
  __syncthreads();

  /* fetch the right data back from the shared memory */
  val = cache[FIX_ME][FIX_ME];

  /* Write the data back to main memory */
  i = blockIdx.x * blockDim.y + threadIdx.y;
  j = blockIdx.y * blockDim.x + threadIdx.x;
  B[i * N + j] = val;
}


int 
main(int argc, char** argv)
{
  /* variables */
  int i, j, cnt, cnt_sm;

  const int dataSize = N * N * sizeof(float);

  /* input and output matrices on host */
  /* output */
  float* B_h_ = (float*) malloc (dataSize);
  float* B_h_sm = (float*) malloc (dataSize);
  /* input */
  float* A_h_ = (float*) malloc (dataSize);
  /* input and output matrices on device */
  float* B_d_, *B_d_sm; 
  float* A_d_; 
  /* reference array */
  float* B_ref_ = (float*) malloc (dataSize);

  struct stopwatch_t* timer = NULL;
  long double t_naive, t_sm;

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

  /* seed random number generator */
  srand48(time(NULL));

  /* Initialize array */
  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      A_h_[i * N + j] = drand48();
    }
  }

  /* allocate memory on host for A and B */
  cudaMalloc ((void**)&(A_d_), dataSize);
  cudaMalloc ((void**)&(B_d_), dataSize);
  cudaMalloc ((void**)&(B_d_sm), dataSize);
	
  /* copy data from host to device */
  cudaMemcpy (A_d_, A_h_, dataSize, cudaMemcpyHostToDevice);


  /* execute kernel */
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks((N / BLOCK_SIZE), (N / BLOCK_SIZE));

  stopwatch_start (timer);
  matTranspose_naive <<<numBlocks, threadsPerBlock>>> (B_d_, A_d_);
  cudaThreadSynchronize ();
  t_naive = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute naive GPU transpose kernel: %Lg secs\n",
           t_naive);


  stopwatch_start (timer);
  matTranspose_sm <<<numBlocks, threadsPerBlock>>> (B_d_sm, A_d_);
  cudaThreadSynchronize ();
  t_sm = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute optimized GPU transpose kernel: %Lg secs\n",
           t_sm);

  /* copy data back from device to host */
  cudaMemcpy (B_h_, B_d_, dataSize, cudaMemcpyDeviceToHost);
  cudaMemcpy (B_h_sm, B_d_sm, dataSize, cudaMemcpyDeviceToHost);

  /* compute reference array */
  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      B_ref_[j * N + i] = A_h_[i * N + j];
    }
  }

  /* check correctness */
  cnt = 0;
  cnt_sm = 0;
  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      if(B_ref_[i * N + j] != B_h_[i * N + j]) {
        cnt++;
      }
      if(B_ref_[i * N + j] != B_h_sm[i * N + j]) {
        cnt_sm++;
      }
    }
  }
  if((cnt > 0) || (cnt_sm > 0)) { 
    printf("ERROR: %d out of %d elements are incorrect for naive\n", 
           cnt, (N * N));
    printf("ERROR: %d out of %d elements are incorrect for SM version\n", 
           cnt_sm, (N * N));
  } else {
    printf("SUCCESS\n");
  }

  return 0;
}
