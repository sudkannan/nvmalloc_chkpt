/******************************************************************************
* FILE: mergesort.c
* DESCRIPTION:  
*   The master task distributes an array to the workers in chunks, zero pads for equal load balancing
*   The workers sort and return to the master, which does a final merge
******************************************************************************/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include "util_func.h"
#include <sys/time.h>
#include <math.h>
#include <sched.h>
#include "jemalloc/jemalloc.h"
#include "nv_map.h"
#include <pthread.h>
#include "nv_rmtckpt.h"
#include <signal.h>
#include <sys/queue.h>
#include <sys/resource.h>
#include "error_codes.h"
#include "nv_def.h"
#include "np_malloc.h"
#include "c_io.h"

//#define _CHKPTIF_DEBUG
//#define _NONVMEM
#define _NVMEM
//#define IO_FORWARDING
//#define LOCAL_PROCESSING
#define FREQUENCY 1
//#define FREQ_MEASURE
#define MAXVARLEN 30

#define BASEID 300

long simulationtime(struct timeval start,
			 struct timeval end );

int thread_init = 0, set_affinity=0;
double startT, stopT;
double startTime;
unsigned long total_bytes=0;
int iter_count =0;
struct timeval g_start, g_end;
struct timeval g_chkpt_inter_strt, g_chkpt_inter_end;
int g_mypid=0;
int mallocid = 0;

#ifdef FREQ_MEASURE
//For measurement
double g_iofreq_time=0;
#endif

pthread_mutex_t precommit_mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t precommit_cond = PTHREAD_COND_INITIALIZER;
int precommit=0;
int curr_chunkid =0;
int prev_chunkid =0;


void *nv_jemalloc(size_t size, rqst_s *rqst) {

	return je_malloc_((size_t)size, rqst);
}

void* nvread_(char *var, int id)
{
    void *buffer = NULL;
    rqst_s rqst;

    id = BASEID;
    rqst.pid = id+1;
    rqst.var_name = (char *)malloc(MAXVARLEN);
    memcpy(rqst.var_name,var,MAXVARLEN);
    rqst.var_name[MAXVARLEN] = 0;
	rqst.no_dram_flg = 1;
    fprintf(stdout,"proc %d var %s\n",id, rqst.var_name);
    buffer = nv_map_read(&rqst, NULL);
    //buffer = rqst.log_ptr;

    if(rqst.var_name)
        free(rqst.var_name);

    return buffer;
}


void* nvalloc_( size_t size, char *var, int id)
{
	void *buffer = NULL;
	rqst_s rqst;

	id = BASEID;

#ifdef ENABLE_RESTART
	/*buffer = nvread(var, id);
	if(buffer) {
		//fprintf(stdout, "nvread succedded \n");
		//return malloc(size);
		return buffer;
	 }*/
#endif

	g_mypid = id+1;
	rqst.id = ++mallocid;
	rqst.pid = id+1;
	rqst.commitsz = size;
	rqst.no_dram_flg = 1;
	rqst.var_name = (char *)malloc(MAXVARLEN);
	memcpy(rqst.var_name,var,MAXVARLEN);
	rqst.var_name[MAXVARLEN] = 0;
	je_malloc_((size_t)size, &rqst);
	//buffer = rqst.log_ptr;
	buffer = rqst.nv_ptr;
	assert(buffer);

	/*fprintf(stdout, "allocated nvchunk %s,"
		"size %u addr: %lu\n,",
		 rqst.var_name,
		 size,
		 (unsigned long)buffer);*/

	if(rqst.var_name)
	  free(rqst.var_name);
	return buffer;
}




