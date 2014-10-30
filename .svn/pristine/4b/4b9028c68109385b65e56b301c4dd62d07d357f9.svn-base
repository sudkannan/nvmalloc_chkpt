/*
 * Copyright (C) 2010. See COPYRIGHT in top-level directory.
 */

#ifdef _ARMCI_CHECKPOINT

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
extern "C" {
#include <armci.h>
}
#include <iostream>
#include <sys/time.h>
#include "util_func.h"
#include "checkpoint.h"
#include "armci_checkpoint.h"

//#define _BW_PROFILING

#define XDIM 1024 
#define YDIM 1024
#define ITERATIONS 10


int grp_nproc;
ARMCI_Group  g_world, my_grp;
int grp_my_rank;

int invoke_barrier() {

	ARMCI_Barrier();
	//ARMCI_Access_begin(buffer[rank]);
}


void** create_memory(int nranks, int myrank, size_t bytes) {

	armci_size_t u_bytes;	
	void **rmt_armci_ptr;
	struct timeval start,end;
	long armci_malloc_time;

	if(myrank == 0)
		gettimeofday(&start,NULL);
	

	rmt_armci_ptr = (void **) calloc(sizeof(void *), nranks);
	assert(rmt_armci_ptr);
 	u_bytes = bytes;
	ARMCI_Malloc(rmt_armci_ptr, u_bytes);
	ARMCI_Barrier();
    //ARMCI_Access_begin(rmt_armci_ptr[myrank]);

	if(myrank == 0){
		gettimeofday(&end,NULL);
	  	armci_malloc_time =simulation_time(start, end );
	    fprintf(stdout,"armci_malloc_time %ld \n",armci_malloc_time);
	}
	
    return rmt_armci_ptr;
}

void** group_create_memory(int nranks, int myrank, size_t bytes) {

	armci_size_t u_bytes;	
	void **rmt_armci_ptr;
	struct timeval start,end;
	long armci_malloc_time;

	if(myrank == 0)
		gettimeofday(&start,NULL);
	
	rmt_armci_ptr = (void **) malloc(sizeof(void *) *nranks);
	assert(rmt_armci_ptr);
 	u_bytes = bytes;

	ARMCI_Malloc_group(rmt_armci_ptr, u_bytes, &my_grp);
	ARMCI_Barrier();
	/*if(myrank == 0){
		gettimeofday(&end,NULL);
	  	armci_malloc_time =simulation_time(start, end );
	    fprintf(stdout,"armci_malloc_time %ld \n",armci_malloc_time);
	}*/
    return rmt_armci_ptr;
}


int coordinate_chunk(int chunk, int mypeer, int myrank) {

	MPI_Status stat;
	int recv_chunk;

	if(myrank == 0){
		 MPI_Send(&chunk, 1, MPI_INT, mypeer, 1, MPI_COMM_WORLD);
	 }else {
		MPI_Recv(&recv_chunk, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &stat);
	}

	return recv_chunk;
}


int create_group ( int *members, int cnt, int myrank,  int numrank) {

	  ARMCI_Group_get_world(&g_world);

      ARMCI_Group_create_child(cnt, members, &my_grp, &g_world);

	  ARMCI_Group_rank(&my_grp, &grp_my_rank);
	  ARMCI_Group_size(&my_grp, &grp_nproc);

#ifdef ARMCI_DEBUG
	  fprintf(stdout,"myrank %d, mygroup rank %d "
				"num group proc %d\n",
				myrank, grp_my_rank,grp_nproc);	
	 for(int i=0; i < cnt; i++)
		 fprintf(stdout,"member[%d]:%d\n",i,members[i]);

	fprintf(stdout,"====================\n");
#endif
}



#ifdef _BW_PROFILING
int send_cnt = 0;
#endif

int armci_remote_memcpy(int myrank, int my_peer,
                        void **rmt_armci_ptr, void *src, size_t bytes){

     int group_peer = 0;
	 struct timeval start, end;



	 if(grp_my_rank == 0)
			group_peer = grp_my_rank + 1;
	   else
			group_peer = grp_my_rank - 1;

#ifdef _BW_PROFILING
	 gettimeofday(&start,NULL);	
#endif

	ARMCI_Put(rmt_armci_ptr[grp_my_rank], 
				rmt_armci_ptr[group_peer], 
				bytes,my_peer);
	/*ARMCI_Put(src, 
				(void *)rmt_armci_ptr[my_peer], 
				bytes, my_peer);*/

#ifdef _BW_PROFILING
	send_cnt++;
	gettimeofday(&end,NULL);
	add_bw_timestamp(send_cnt,start, end, bytes);
#endif 
	//ARMCI_Barrier();
	 return 0;
}




int main1(int argc, char **argv) {
    int i, j, rank, nranks, peer, bufsize, errors;
    double **buffer, *src_buf;
    int count[2], src_stride, trg_stride, stride_level;
    int members[2];
	int cnt=2;
	int mypeer;
    size_t bytes = 0;
    void **mybuffer;

    MPI_Init(&argc, &argv);
    ARMCI_Init();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    buffer = (double **) malloc(sizeof(double *) * nranks);


	if(rank%2 == 0) {
		mypeer = rank +1;
		members[0] = rank;
		members[1] = mypeer;

	}else {
		mypeer = rank -1;
		members[0] = mypeer;
		members[1] = rank;
	}

    create_group (members, cnt, rank, nranks);	

	bytes = 1024*1024*10;
	mybuffer = group_create_memory(nranks, rank,  bytes);

    armci_remote_memcpy(rank, mypeer,mybuffer, mybuffer[rank],bytes);


    ARMCI_Finalize();
    MPI_Finalize();

    if (errors == 0) {
      printf("%d: Success\n", rank);
      return 0;
    } else {
      printf("%d: Fail\n", rank);
      return 1;
    }
}

#endif
