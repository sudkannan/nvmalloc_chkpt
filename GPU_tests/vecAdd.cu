#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "timer.h"

#include "checkpoint.h"

int restart = 1;

/*
 * **CUDA KERNEL** 
 * 
 * Compute the sum of two vectors 
 *   C[i] = A[i] + B[i]
 * 
 */
__global__ void vecAdd(float* a, float* b, float* c) {

  /* Calculate index for this thread */
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Compute the element of C */
  c[i] = a[i] + b[i];
	printf("cuda_c[%d]:%f\n",i,c[i]);

}

__global__ void printKernel(float* a, float* b, float* c) {

  /* Calculate index for this thread */
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for(i = 0; i < 1024; i++) {
	printf("cuda_c[%d]:%f\n",i,c[i]);
  }
}



void compute_vec_add(int N, float *a, float* b, float *c); 

/*
 * 
 * Host code to drive the CUDA Kernel
 * 
 */
int main(int argc, char *argv[]) { 

  float *d_a, *d_b, *d_c;
  float *h_a, *h_b, *h_c, *h_temp;
  int i; 
  int N = 1024 * 1024;

  struct stopwatch_t* timer = NULL;
  struct stopwatch_t* prog_timer = NULL;
  long double fullprog_t = 0;

  long double t_pcie_htd, t_pcie_dth, t_kernel, t_cpu;


  restart = atoi(argv[1]);


  /* Setup timers */
  stopwatch_init ();

  timer = stopwatch_create ();
  prog_timer = stopwatch_create ();	


  /*
    Create the vectors
  */
  h_a = (float *) malloc(sizeof(float) * N);
  h_b = (float *) malloc(sizeof(float) * N);
  h_c = (float *) malloc(sizeof(float) * N);

  /*
    Set the initial values of h_a, h_b, and h_c
  */
  for (i=0; i < N; i++) {
    h_a[i] = (float) (rand() % 100) / 10.0;
    h_b[i] = (float) (rand() % 100) / 10.0;
    h_c[i] = 0.0;
  }

  /*
    Run N/256 blocks of 256 threads each
  */
  dim3 GS (N/256, 1, 1);
  dim3 BS (256, 1, 1);

  /*
    Allocate space on the GPU
  */

  stopwatch_start (prog_timer);	

  if( restart == 0){
	  printf("not restarting \n");
	  CUDA_CHECK_ERROR(chk_cudaMalloc((void **)&d_a, sizeof(float) * N,(char *)"d_a"));
	  CUDA_CHECK_ERROR(chk_cudaMalloc((void **)&d_b, sizeof(float) * N, "d_b"));
	  CUDA_CHECK_ERROR(chk_cudaMalloc((void **)&d_c, sizeof(float) * N, "d_c"));
	  /*
	    Copy d_a and d_b from CPU to GPU
	  */
	  stopwatch_start (timer);
	  CUDA_CHECK_ERROR(cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice));
	  CUDA_CHECK_ERROR(cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice));
	  t_pcie_htd = stopwatch_stop (timer);
	  fprintf (stderr, "Time to transfer data from host to device: %Lg secs\n", 
           t_pcie_htd);
	}else {

	  printf("restarting \n");

	  CUDA_CHECK_ERROR(chk_cudaMalloc_restart((void **)&d_a, sizeof(float) * N,(char *)"d_a"));
	  CUDA_CHECK_ERROR(chk_cudaMalloc_restart((void **)&d_b, sizeof(float) * N, "d_b"));
	  CUDA_CHECK_ERROR(chk_cudaMalloc_restart((void **)&d_c, sizeof(float) * N, "d_c"));

	  fullprog_t = stopwatch_stop(prog_timer);
	  fprintf (stderr, "Time fullprog_t %Lg\n",fullprog_t);

	  goto finish;	
	}


  stopwatch_start (timer);
  vecAdd<<<GS, BS>>>(d_a, d_b, d_c);
  cudaThreadSynchronize ();
  t_kernel = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute GPU kernel: %Lg secs\n", 
           t_kernel);

#ifdef _USECHKPT    
    cudaDeviceSynchronize();
    chk_cudaCommit(NULL, 1);
#endif




  finish:


  /*
    Copy d_cfrom GPU to CPU
  */
  stopwatch_start (timer);
  CUDA_CHECK_ERROR(cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));

  if(restart == 1) {
	  CUDA_CHECK_ERROR(cudaMemcpy(h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost));
	  CUDA_CHECK_ERROR(cudaMemcpy(h_b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost));
   }

  t_pcie_dth = stopwatch_stop (timer);


  fprintf (stderr, "Time to transfer data from device to host: %Lg secs "
			"fullprog_t %Lg\n", t_pcie_dth, fullprog_t);

  //for(int i = 0; i < N; i++) {
   	//fprintf(stdout,"h_a[%d]:%f,h_b[%d]:%f,h_c[%d]:%f\n",i,h_a[i],i,h_b[i],i,h_c[i]);
	//fprintf(stdout,"h_c[%d]:%f\n",i,h_c[i]);
  //}

  /* 
     Double check errors
  */
  h_temp = (float *) malloc(sizeof(float) * N);
  stopwatch_start (timer);
  compute_vec_add (N, h_a, h_b, h_temp);
  t_cpu = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute CPU program: %Lg secs\n", 
           t_cpu);

  int cnt = 0;
  for(int i = 0; i < N; i++) {
    if(abs(h_temp[i] - h_c[i]) > 1e-5) cnt++;
  }
  fprintf(stderr, "number of errors: %d out of %d\n", cnt, N);


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
compute_vec_add(int N, float *a, float* b, float *c) {

    for(int i=0; i < N; i++)
        c[i] = a[i] + b[i];


}


