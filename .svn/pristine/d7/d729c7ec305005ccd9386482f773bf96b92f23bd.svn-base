/******************************************************************************
* FILE: mergesort.c
* DESCRIPTION:  
*   The master task distributes an array to the workers in chunks, zero pads for equal load balancing
*   The workers sort and return to the master, which does a final merge
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include <math.h>
#include <sched.h>
#include "include/nv_map.h"
#include "include/c_io.h"

#ifdef ENABLE_MPI_RANKS 
#include "mpi.h"
#endif

#define ITERATION 10



/*************************************************************/
/*          Example of writing arrays in NVCHPT               */
/*                                                           */
/*************************************************************/
int main (int argc, char ** argv)
{
    char        filename [256];
    int         rank, j;
    int         NY = 100;
    int         *p;
	int 	    *chunk2;
	int mype = 0;
    int base = 0;
	UINT size;
#ifdef ENABLE_MPI_RANKS 
    MPI_Comm    comm = MPI_COMM_WORLD;
#endif
    UINT NX, i; 


	fprintf(stdout, "starting NVM checkpoint test \n");


	NX  = 1024 * 1024 * 50;

#ifdef ENABLE_MPI_RANKS 
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (comm, &rank);
#endif

    if(argc > 1)
	base = atoi(argv[1]);

	mype = base + rank;


		size = NX*sizeof(int);
		fprintf(stdout, "allocating from NVM \n");
		p = (int *) nvalloc_(size,(char *)"zion", mype);
	 	chunk2 =(int *) nvalloc_(size,(char *)"chunk2", mype);
		fprintf(stdout, "finished allocating from NVM \n");


	for (int itr=0; itr < ITERATION; itr++) {   
    	for (i = 0; i < NX; i++) {
        	p[i] = rank + NX * i;
	        //fprintf(stdout, "P[%d]:%d \n",
    	     //      i, p[i]);
		 }
    	 for (i = 0; i < NX; i++) {
        	chunk2[i] = rank + NX * i;
	        //fprintf(stdout, "chunk2[%d]:%d \n",
    	    //        i, chunk2[i]);
		 }
		 fprintf(stdout,"starting checkpoint\n");
		 nvchkpt_all_(&mype);
	}

#ifdef ENABLE_MPI_RANKS 
    MPI_Barrier (comm);
    MPI_Finalize ();
#endif
    return 0;
}




























































