#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "timer.h"

/*
 * **CUDA KERNEL** 
 * 
 * Compute the sum of two matrices 
 *   C[i] = A[i] + B[i]
 * 
 */
__global__ void matAdd(int N, float* a, float* b, float* c) {


	 /* Calculate array index for this thread */
	 int i = blockIdx.y * blockDim.y + threadIdx.y;
	 int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	 int idx = i * gridDim.x *  blockDim.x + j;

	  c[idx] = 	a[idx] + b[idx];

}

void compute_mat_add(int N, float *a, float* b, float *c); 

/*
 * 
 * Host code to drive the CUDA Kernel
 * 
 */
int main() { 

  float *d_a, *d_b, *d_c;
  float *h_a, *h_b, *h_c, *h_temp;
  int i; 
  int N = 256 * 1;

  struct stopwatch_t* timer = NULL;
  long double t_pcie_htd, t_pcie_dth, t_kernel, t_cpu;

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

  /*
    Create the matrices 
  */
  h_a = (float *) malloc(sizeof(float) * N * N);
  h_b = (float *) malloc(sizeof(float) * N * N);
  h_c = (float *) malloc(sizeof(float) * N * N);

  /*
    Set the initial values of h_a, h_b, and h_c
  */
  for (i=0; i < N * N; i++) {
    h_a[i] = (float) (rand() % 100) / 10.0;
    h_b[i] = (float) (rand() % 100) / 10.0;
    h_c[i] = 0.0;
  }


  /*
    Allocate space on the GPU
  */
  CUDA_CHECK_ERROR(cudaMalloc(&d_a, sizeof(float) * N * N));
  CUDA_CHECK_ERROR(cudaMalloc(&d_b, sizeof(float) * N * N));
  CUDA_CHECK_ERROR(cudaMalloc(&d_c, sizeof(float) * N * N));

  /*
    Copy d_a and d_b from CPU to GPU
  */
  stopwatch_start (timer);
  CUDA_CHECK_ERROR(cudaMemcpy(d_a, h_a, sizeof(float) * N * N, 
                              cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_b, h_b, sizeof(float) * N * N, 
                              cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_c, h_c, sizeof(float) * N * N, 
                              cudaMemcpyHostToDevice));

  t_pcie_htd = stopwatch_stop (timer);
  t_pcie_htd = stopwatch_stop (timer);
  fprintf (stderr, "Time to transfer data from host to device: %Lg secs\n", 
           t_pcie_htd);

  /*
    Run N/256 blocks of 256 threads each
  */
  dim3 GS(N/16, N/16, 1);
  dim3 BS(16, 16, 1);
  //dim3 GS(N*N/256);
  //dim3 BS(256);

  stopwatch_start (timer);
  matAdd<<<GS, BS>>>(N, d_a, d_b, d_c);
  cudaThreadSynchronize ();
  t_kernel = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute GPU kernel: %Lg secs\n", 
           t_kernel);

  /*
    Copy d_cfrom GPU to CPU
  */
  stopwatch_start (timer);
  CUDA_CHECK_ERROR(cudaMemcpy(h_c, d_c, sizeof(float) * N * N, 
                              cudaMemcpyDeviceToHost));
  t_pcie_dth = stopwatch_stop (timer);
  fprintf (stderr, "Time to transfer data from device to host: %Lg secs\n", 
           t_pcie_dth);


  /* 
     Double check errors
  */
  h_temp = (float *) malloc(sizeof(float) * N * N);
  stopwatch_start (timer);
  compute_mat_add (N, h_a, h_b, h_temp);
  t_cpu = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute CPU program: %Lg secs\n", 
           t_cpu);

  int cnt = 0;
  for(int i = 0; i < N * N; i++) {
    if(abs(h_temp[i] - h_c[i]) > 1e-5) cnt++;
  }
  fprintf(stderr, "number of errors: %d out of %d\n", cnt, N * N);


  /*
    Free the device memory
  */
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  /*
    Free the host memory
  */
  free(h_a);
  free(h_b);
  free(h_c);

  /* 
     Free timer 
  */
  stopwatch_destroy (timer);

  if(cnt == 0) {
    printf("\n\nSuccess\n");
  }

}

void
compute_mat_add(int N, float *a, float* b, float *c) {
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      c[i * N + j] = a[i * N + j] + b[i * N + j];
    }
  }
}


