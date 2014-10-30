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

/*************************************************************/
/*          Example of writing arrays in NVCHPT               */
/*                                                           */
/*************************************************************/
int main (int argc, char ** argv)
{
    char        filename [256];
    int         rank, size, i, j;
    int         NX = 10, NY = 100;
    double      t[NX][NY];
    int         *p;
	int mype = 0;
#ifdef ENABLE_MPI_RANKS
    MPI_Comm    comm = MPI_COMM_WORLD;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (comm, &rank);
#endif

	mype = atoi(argv[1]);
	p = (int *)nvread_((char *)"zion", mype);
    assert(p);

 	int *chunk2 =(int *) nvread_((char *)"chunk2", mype);
	assert(chunk2);

    for (i = 0; i < NX; i++)
        fprintf(stdout, "P[%d]:%d chunk2[%d]:%d\n",
				i, p[i], i, chunk2[i]);

#ifdef ENABLE_MPI_RANKS
    MPI_Barrier (comm);
    MPI_Finalize ();
#endif
    return 0;
}





























































