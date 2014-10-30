
/*
 *  malloc-test
 *  cel - Thu Jan  7 15:49:16 EST 1999
 *
 *  Benchmark libc's malloc, and check how well it
 *  can handle malloc requests from multiple threads.
 *
 *  Syntax:
 *  malloc-test [ size [ iterations [ thread count ]]]
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include "include/nv_map.h"
#include "include/c_io.h"

#ifdef ENABLE_MPI_RANKS 
	#include "mpi.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>



#define NULL 0

#define USECSPERSEC 1000000
#define pthread_attr_default NULL
#define MAX_THREADS 2

#define BASE_PROC_ID 10000

unsigned int procid;
void * dummy(unsigned);
static unsigned size = 1024 * 1024 *100;
static unsigned iteration_count = 100; 

#define OUTFILENAME "/tmp/file1"
#define OUTBYTES 1024*1024*1024


 int  setup_map_file(char *filepath, ULONG bytes)
 {
    int result;
        int fd;

    fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600);
    if (fd == -1) {
        perror( filepath);
        exit(EXIT_FAILURE);
    }

    result = lseek(fd,bytes,  SEEK_SET);
    if (result == -1) {
        close(fd);
        fprintf(stdout,"Error calling lseek() to 'stretch' the file %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    result = write(fd, "", 1);
    if (result != 1) {
        close(fd);
        fprintf(stdout,"Error writing last byte of the file %s", filepath);
        exit(EXIT_FAILURE);
    }
    return fd;
}

//#if defined(__cplusplus)
//xtern "C"
//#endif
void *run_test(void* val)
{
	register unsigned int i;
	register unsigned request_size = size;
	register unsigned total_iterations = iteration_count;
	struct timeval start, end, null, elapsed, adjusted;
	int rank = 0;
	int fd = -1;
	void *addr[200];

#ifdef ENABLE_MPI_RANKS 
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	//fprintf(stderr,"rank %d \n",rank);
#endif
 //#ifdef USE_NVMALLOC
    rqst_s rqst;
//#endif
	/*
	 * Time a null loop.  We'll subtract this from the final
	 * malloc loop results to get a more accurate value.
	 */
	null.tv_sec = end.tv_sec - start.tv_sec;
	null.tv_usec = end.tv_usec - start.tv_usec;
	if (null.tv_usec < 0) {
		null.tv_sec--;
		null.tv_usec += USECSPERSEC;
	}

	fd = setup_map_file(OUTFILENAME, OUTBYTES);
	/*
	 * Run the real malloc test
	 */
	gettimeofday(&start, NULL);
	int j =0;
	for (j = 0; j< total_iterations; j++) {

		register void * buf;
		rqst.id = j+1;
		rqst.var_name = NULL;
		rqst.pid = rank+1+ BASE_PROC_ID;
		size = 1024*1024*100;

		void *temp;

		if(j == 0)
			temp = 0;
		else
			temp = addr[j-1];

       addr[j] = mmap(temp,size,PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
		//nv_jemalloc(size,&rqst);
		char *ptr = (char *)addr[j];
		fprintf(stdout, "ptr %lu \n",(unsigned long)ptr);
		for (int k = 0; k < 10; k++) 
			for (i = 0; i < size; i++) {
			ptr[i] = 'a';
		}
		//free(buf);
	}

	gettimeofday(&end, NULL);

	elapsed.tv_sec = end.tv_sec - start.tv_sec;
	elapsed.tv_usec = end.tv_usec - start.tv_usec;
	if (elapsed.tv_usec < 0) {
		elapsed.tv_sec--;
		elapsed.tv_usec += USECSPERSEC;
	}

	/*
	 * Adjust elapsed time by null loop time
	 */
	adjusted.tv_sec = elapsed.tv_sec - null.tv_sec;
	adjusted.tv_usec = elapsed.tv_usec - null.tv_usec;
	if (adjusted.tv_usec < 0) {
		adjusted.tv_sec--;
		adjusted.tv_usec += USECSPERSEC;
	}
	printf("Thread %d adjusted timing: %d.%06d seconds for %d requests"
		" of %d bytes.\n", pthread_self(),
		adjusted.tv_sec, adjusted.tv_usec, total_iterations,
		request_size);

	//pthread_exit(NULL);
}



int main(int argc, char *argv[])
{
	unsigned i;
	unsigned thread_count = 1;
	pthread_t thread[MAX_THREADS];

#ifdef ENABLE_MPI_RANKS	
	MPI_Init (&argc, &argv);	
#endif

	run_test(NULL);
	/*
	 * Parse our arguments
	 */
	/*switch (argc) {

	case 5:
		// size, iteration count, and thread count were specified */
		/*thread_count = atoi(argv[4]);
		if (thread_count > MAX_THREADS) thread_count = MAX_THREADS;

	case 4:
		procid = atoi(argv[3]);

	case 3:
		// size and iteration count were specified; others default */
		/*iteration_count = atoi(argv[2]);
	case 2:
		// size was specified; others default */
		/*size = atoi(argv[1]);
	case 1:
		// use default values */
		/*break;
	default:
		printf("Unrecognized arguments.\n");
		exit(1);
	}*/

	/*
	 * Invoke the tests
	 */
	/*printf("Starting test...\n");
	for (i=1; i<=thread_count; i++)
		if (pthread_create(&(thread[i]), pthread_attr_default,&run_test, NULL))
			printf("failed.\n");*/

	/*
	 * Wait for tests to finish
	 */
	/*for (i=1; i<=thread_count; i++)
		pthread_join(thread[i], NULL);*/
#ifdef ENABLE_MPI_RANKS
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	exit(0);
}

void * dummy(unsigned i)
{
	return NULL;
}


