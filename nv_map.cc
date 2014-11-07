#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <strings.h>
#include <time.h>
#include <assert.h>
#include <inttypes.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

#include <iostream>
#include <string>
#include <map>
#include <queue>
#include <list>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <sstream>

#include "nv_map.h"
#include "nv_def.h"
#include "checkpoint.h"
#include "util_func.h"
#include "time_delay.h"
#include "jemalloc/jemalloc.h"
#include "LogMngr.h"
#include "nv_transact.h"
#include "nv_stats.h"
#include "nv_debug.h"
#include "malloc_hook.h"

#include "gtthread.h"
#include "gtthread_spinlocks.h"

//#include "google_hash.h"

#ifdef _USEPIN
#include "pin_mapper.h"
#endif

using namespace std;

//using namespace snappy;

///////////////////////glob vars///////////////////////////////////////////////////
//checkpoint related code
pthread_mutex_t chkpt_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t dataPresentCondition = PTHREAD_COND_INITIALIZER;
int mutex_set = 0;
int dataPresent = 0;
int local_chkpt_cnt = 0;
int dummy_var = 0;
int proc_list_init = 0;
ULONG proc_map_start;
int g_file_desc = -1;

nvarg_s nvarg;
rbtree map_tree;
rbtree proc_tree;
long chkpt_itr_time;
std::map<int, int> fault_chunk, fault_hist;
std::map<int, int>::iterator fault_itr;
int stop_history_coll, checpt_cnt;

#ifdef _USE_FAULT_PATTERNS
int chunk_fault_lst_freeze;
#endif

/*contains a list of all pointers allocated
 during a session. This map is loaded during
 restart, to know what where the previous
 pointers and where did they map to*/
std::map<void *, chunkobj_s*> pntr_lst;
rbtree pntrlst_tree;
static int pntrlst_tree_init;

/*enabled if _USE_CACHEFLUSH
 * is enabled
 *  */
uint8_t useCacheFlush;
uint8_t enable_debug;
uint8_t enable_optimze;
uint8_t enable_shadow;
uint8_t enable_trans;
uint8_t enable_undolog;
uint8_t enable_stats;

//#ifdef _NVRAM_OPTIMIZE
#define NUM_MMAP_CACHE_CNT 128
proc_s* prev_proc_obj = NULL;
UINT prev_proc_id;
ULONG mmap_strt_addr_cache[NUM_MMAP_CACHE_CNT];
UINT mmap_size_cache[NUM_MMAP_CACHE_CNT];
ULONG mmap_ref_cache[NUM_MMAP_CACHE_CNT];
UINT next_mmap_entry;
uint8_t use_map_cache;
//#endif

extern std::unordered_map <unsigned long, size_t> allocmap;
extern std::unordered_map <unsigned long, size_t> alloc_prot_map;
//std::unordered_map<void *, size_t> fault_stat;
std::unordered_map<unsigned long, size_t> fault_stat;
extern std::map <void *, size_t> life_map;
void print_stat();

///////////////////////MACROS///////////////////////////////////////////////////

#ifdef _ENABLE_SWIZZLING
#define handle_error(msg) \
		do { perror(msg); exit(EXIT_FAILURE); } while (0)

static void handler(int sig, siginfo_t *si, void *unused)
{

	fprintf(stdout,"recvd seg fault %lu\n", (unsigned long)si->si_addr);
	load_valid_addr(&si->si_addr);
}

static void initseghandling()
{
	struct sigaction sa;
	struct sched_param param;
	sa.sa_flags = SA_SIGINFO;
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = handler;
	//if (sigaction(SIGSEGV, &sa, NULL) == -1)
	//  handle_error("sigaction");
}
#endif

int file_error(char *filename) {

	fprintf(stdout, "error opening file %s \n", filename);
	return 0;
}

int CompRange(node key_node, void* a, void* b) {

	struct mmapobj_nodes *mmapobj_struct =
			(struct mmapobj_nodes *) key_node->value;

	ULONG a_start_addr = (ULONG) a;
	ULONG b_start_addr = (ULONG) b;

	if ((a_start_addr > b_start_addr)
			&& (mmapobj_struct->end_addr > a_start_addr)) {
		return (0);
	}

	if (a_start_addr > b_start_addr)
		return (1);
	if (a_start_addr < b_start_addr)
		return (-1);
	return (0);
}

int IntComp(node n, void* a, void* b) {
	if ((uintptr_t) a > (uintptr_t) b)
		return (1);
	if ((uintptr_t) a < (uintptr_t) b)
		return (-1);
	return (0);
}

#ifdef _USEPIN
int num_maps;
void init_pin() {
    //CreateSharedMem();
}
#endif

void *mmap_wrap(void *addr, size_t size, int mode, int prot, int fd,
		size_t offset, nvarg_s *s) {

	void* nvmap;
	pthread_mutex_lock( &chkpt_mutex );
#ifdef _USE_FAKE_NVMAP
	nvmap = (void *)mmap(addr, size, mode, prot, fd, 0);
#else
	//fprintf(stdout,"calling mmap_wrap sys nvmmap\n");
	//nvmap = (void *)mmap(addr, size, mode, prot, -1, 0);
	nvmap = (void *)syscall(__NR_nv_mmap_pgoff,addr,size,mode,prot, s);
	assert(nvmap);
	//memset(nvmap,0, size);
#endif

#ifdef _USEPIN
    //creates shared memory. if shared memory already created
    // then returns pointer
    init_pin();
    num_maps++;
    //printf("Writing line %lu %lu %d\n", nvmap,nvmap+size,num_maps);
    Writeline((unsigned long)nvmap, (unsigned long)nvmap+size);
#endif

	pthread_mutex_unlock(&chkpt_mutex);

	return nvmap;
}

int init_chunk_tree(mmapobj_s *mmapobj) {

	assert(mmapobj);
	mmapobj->chunkobj_tree = rbtree_create();
	mmapobj->chunk_tree_init = 1;
	return 0;
}

int copy_chunkoj(chunkobj_s *dest, chunkobj_s *src) {
	assert(dest);
	dest->pid = src->pid;
	dest->chunkid = src->chunkid;
	dest->length = src->length;
	dest->vma_id = src->vma_id;
	dest->offset = src->offset;
	return 0;
}

//RBtree code ends
int initialize_mmapobj_tree(proc_s *proc_obj) {

	assert(proc_obj);
	if (!proc_obj->mmapobj_tree) {
		proc_obj->mmapobj_tree = rbtree_create();
		assert(proc_obj->mmapobj_tree);
		proc_obj->mmapobj_initialized = 1;
	}
	return 0;
}

int copy_mmapobj(mmapobj_s *dest, mmapobj_s *src) {

	assert(dest);
	dest->vma_id = src->vma_id;
	dest->length = src->length;
	dest->proc_id = src->proc_id;
	dest->offset = src->offset;
	dest->numchunks = src->numchunks;
	return 0;
}

/*creates mmapobj object.sets object variables to app. value*/
static mmapobj_s* create_mmapobj(rqst_s *rqst, ULONG curr_offset,
		proc_s* proc_obj, ULONG data_addr) {

	mmapobj_s *mmapobj = NULL;
	ULONG addr = 0;

	assert(rqst);
	assert(proc_obj);
	assert(proc_map_start);

	addr = proc_map_start;
	addr = addr + proc_obj->meta_offset;
	mmapobj = (mmapobj_s*) addr;
	proc_obj->meta_offset += sizeof(mmapobj_s);

	mmapobj->vma_id = rqst->id;
	mmapobj->length = rqst->bytes;
	mmapobj->proc_id = rqst->pid;
	mmapobj->offset = curr_offset;
	mmapobj->data_addr = data_addr;
	rqst->id = BASE_METADATA_NVID + mmapobj->vma_id;
	rqst->bytes = MMAP_METADATA_SZ;

#ifdef _USE_FAKE_NVMAP
	char file_name[256];
	char fileid_str[64];
	bzero(file_name,256);
	generate_file_name((char *) PROCMAPMETADATA_PATH,rqst->pid, file_name);
	sprintf(fileid_str, "%d", rqst->id);
	strcat(file_name,"_");
	strcat(file_name, fileid_str);
	int fd = setup_map_file(file_name, rqst->bytes);
	assert(fd);
	addr = (ULONG) mmap(0, rqst->bytes,
			PROT_NV_RW, MAP_SHARED, fd, 0);
	//fprintf(stdout,"creating mmap obj %s of size %u\n",
	//		file_name,rqst->bytes);
#else
	addr = (ULONG) (map_nvram_state(rqst));
#endif

	assert((void *)addr != MAP_FAILED);
	mmapobj->strt_addr = addr;
	init_chunk_tree(mmapobj);
	assert(mmapobj->chunkobj_tree);

	return mmapobj;
}

static void update_chunkobj(rqst_s *rqst, mmapobj_s* mmapobj,
		chunkobj_s *chunkobj) {

	assert(chunkobj);
	chunkobj->pid = rqst->pid;
	chunkobj->chunkid = rqst->id;
	chunkobj->length = rqst->bytes;
	chunkobj->vma_id = mmapobj->vma_id;
	chunkobj->offset = rqst->offset;
	//chunkobj->mmapobj = mmapobj;
	chunkobj->nv_ptr = rqst->nv_ptr;
#ifdef _USE_TRANSACTION
	chunkobj->dirty = 0;
#endif
	DEBUG_CHUNKOBJ(chunkobj);

#ifdef _USE_SHADOWCOPY
	//chunkobj->logcpy = 0;
	chunkobj->log_ptr = rqst->log_ptr;
	chunkobj->logptr_sz = rqst->logptr_sz;
	assert(chunkobj->log_ptr);
	//assert(chunkobj->logptr_sz);
#endif
}

/*creates mmapobj object.sets object variables to app. value*/
static chunkobj_s* create_chunkobj(rqst_s *rqst, mmapobj_s* mmapobj) {

	chunkobj_s *chunkobj = NULL;
	ULONG addr = 0;
	UINT mapoffset = 0;

	/*perform null checks*/
	assert(rqst);
	addr = mmapobj->strt_addr;
	assert(addr);

	/*update all member varialbes of chunk struct*/
	mapoffset = mmapobj->meta_offset;
	addr = addr + mapoffset;
	chunkobj = (chunkobj_s*) addr;
	mapoffset += sizeof(chunkobj_s);
	mmapobj->meta_offset = mapoffset;
	update_chunkobj(rqst, mmapobj, chunkobj);

#ifdef _USE_TRANSACTION
	//initialize the chunk dirty bit
	if (enable_trans) {
		chunkobj->dirty = 0;
	}
#endif

#ifdef _NVSTATS
	if(enable_stats) {
		add_stats_chunk(rqst->pid, rqst->bytes);
	}
#endif
	return chunkobj;
}

/*Function to return the process object to which mmapobj belongs
 @ mmapobj: process to which the mmapobj belongs
 @ return: process object
 */
proc_s* get_process_obj(mmapobj_s *mmapobj) {

	if (!mmapobj) {
		fprintf(stdout, "get_process_obj: mmapobj null \n");
		return NULL;
	}
	return mmapobj->proc_obj;
}

chunkobj_s* find_chunk(mmapobj_s *t_mmapobj, unsigned int chunkid) {

	chunkobj_s *chunkobj = NULL;
	if (t_mmapobj) {
		if (t_mmapobj->chunkobj_tree) {
			chunkobj = (chunkobj_s *) rbtree_lookup(t_mmapobj->chunkobj_tree,
					(void*) chunkid, IntComp);
			if (chunkobj) {
				return chunkobj;
			}
		}
	}
	return NULL;
}

int find_vmaid_from_chunk(rbtree_node n, unsigned int chunkid) {

	int ret = 0;
	mmapobj_s *t_mmapobj = NULL;
	chunkobj_s *chunkobj = NULL;

	//assert(n);
	if (n == NULL) {
		return 0;
	}

	if (n->right != NULL) {
		ret = find_vmaid_from_chunk(n->right, chunkid);
		if (ret)
			return ret;
	}
	t_mmapobj = (mmapobj_s *) n->value;
	chunkobj = find_chunk(t_mmapobj, chunkid);
	if (chunkobj) {
		return t_mmapobj->vma_id;
	}

	if (n->left != NULL) {
		return find_vmaid_from_chunk(n->left, chunkid);
	}

	return 0;
}

/*Function to find the mmapobj.
 @ process_id: process identifier
 @ var:  variable which we are looking for
 */
mmapobj_s* find_mmapobj_from_chunkid(unsigned int chunkid, proc_s *proc_obj) {

	mmapobj_s *t_mmapobj = NULL;
	ULONG vmid_l = 0;

	if (!proc_obj) {
		fprintf(stdout, "could not identify project id \n");
		return NULL;
	}
	/*if mmapobj is not yet initialized do so*/
	if (!proc_obj->mmapobj_initialized) {
		fprintf(stdout, "proc_obj->mmapobj_initialized failed \n");
		initialize_mmapobj_tree(proc_obj);
		return NULL;
	}
	if (proc_obj->mmapobj_tree) {

		//print_tree(proc_obj->mmapobj_tree);
		vmid_l = find_vmaid_from_chunk(proc_obj->mmapobj_tree->root, chunkid);
		if (!vmid_l) {
			goto exit;
		}
		t_mmapobj = (mmapobj_s*) rbtree_lookup(proc_obj->mmapobj_tree,
				(void *) vmid_l, IntComp);
	}
	exit: return t_mmapobj;
}

mmapobj_s* find_mmapobj(UINT vmid_l, proc_s *proc_obj) {

	mmapobj_s *t_mmapobj = NULL;
	if (!proc_obj) {
		fprintf(stdout, "could not identify project id \n");
		return NULL;
	}
	/*if mmapobj is not yet initialized do so*/
	if (!proc_obj->mmapobj_initialized) {
		initialize_mmapobj_tree(proc_obj);
		return NULL;
	}
	t_mmapobj = (mmapobj_s*) rbtree_lookup(proc_obj->mmapobj_tree,
			(void *) vmid_l, IntComp);
	return t_mmapobj;
}

/*add the mmapobj to process object*/
static int add_mmapobj(mmapobj_s *mmapobj, proc_s *proc_obj) {

	if (!mmapobj)
		return 1;

	if (!proc_obj)
		return -1;

	if (!proc_obj->mmapobj_initialized)
		initialize_mmapobj_tree(proc_obj);

	//RB tree code
	assert(proc_obj->mmapobj_tree);
	rbtree_insert(proc_obj->mmapobj_tree, (void*) mmapobj->vma_id, mmapobj,
			IntComp);
	//set the process obj to which mmapobj belongs
	mmapobj->proc_obj = proc_obj;

	if (use_map_cache)
		add_to_mmap_cache(mmapobj);
	/*
	 * When ever a mmap obj is added flush to
	 * NVM
	 */
	if (useCacheFlush) {
		flush_cache(mmapobj, sizeof(mmapobj_s));
	}
	return 0;
}

/*add the mmapobj to process object*/
static int add_chunkobj(mmapobj_s *mmapobj, chunkobj_s *chunk_obj) {

	if (!mmapobj)
		return -1;

	if (!chunk_obj)
		return -1;

	if(!chunk_obj->length){
		DEBUG_MMAPOBJ_T(mmapobj);
		//DEBUG_CHUNKOBJ_T(chunk_obj);
		assert(chunk_obj->length);
	}
	/*we flush the newly created
	 * chunk object*/
	if (useCacheFlush) {
		if (chunk_obj) {
			flush_cache((void*) chunk_obj, sizeof(chunkobj_s));
		}
	}
	/*DEBUG("add_chunkobj: chunkid %d"
			"chunk_tree_init %d \n",
			(void*)chunk_obj->chunkid,
			mmapobj->chunk_tree_init);*/
	//OPTIMIZE CHANGE
	if (!mmapobj->chunk_tree_init){
		init_chunk_tree(mmapobj);
	}
	assert(mmapobj->chunkobj_tree);
	rbtree_insert(mmapobj->chunkobj_tree, (void*)(chunk_obj->chunkid),
			(void*)chunk_obj, IntComp);
	//set the process obj to which mmapobj belongs
	//chunk_obj->mmapobj = mmapobj;
	return 0;
}


int map_to_get_chunk_list (mmapobj_s *mmapobj, rqst_s *rqst) {

#ifdef _USE_FAKE_NVMAP
	char file_name[256];
	char fileid_str[64];
	FILE *fp = NULL;
	int fd = -1;

	bzero(file_name,256);
	generate_file_name((char *) PROCMAPMETADATA_PATH,rqst->pid, file_name);
	sprintf(fileid_str, "%d", rqst->id);
	strcat(file_name,"_");
	strcat(file_name, fileid_str);
	fp = fopen(file_name, "a+");
	assert(fp);

	fd = fileno(fp);
	mmapobj->strt_addr = (ULONG) mmap(0, rqst->bytes,
			PROT_NV_RW, MAP_SHARED, fd, 0);

#else
	mmapobj->strt_addr = (ULONG)map_nvram_state(rqst);
#endif
	return 0;
}

int restore_chunk_objs(mmapobj_s *mmapobj, int perm) {

	rqst_s rqst, datarq;
	chunkobj_s *nv_chunkobj = NULL;
	chunkobj_s *chunkobj = NULL;
	void *mem = NULL;
	int idx = 0;
	void *addr = NULL;
	unsigned long addr_l = 0;

	assert(mmapobj);
	rqst.id = BASE_METADATA_NVID + mmapobj->vma_id;
	rqst.pid = mmapobj->proc_id;
	rqst.bytes = MMAP_METADATA_SZ;
	mmapobj->chunk_tree_init = 0;

	/*this will map the region with chunk metadata*/
	map_to_get_chunk_list (mmapobj,&rqst);

	assert((void *)mmapobj->strt_addr != MAP_FAILED);
	assert(mmapobj->strt_addr);

	addr = (void *)mmapobj->strt_addr;
	mmapobj->mmap_offset=0;

	//Now MAP for data
	datarq.id = mmapobj->vma_id;
	datarq.pid = mmapobj->proc_id;
	datarq.bytes = mmapobj->length;

#ifdef _USE_FAKE_NVMAP
	char file_name[MAX_FPATH_SZ];
	char fileid_str[64];
	FILE *fp = NULL;
	int fd = -1;

	bzero(file_name,256);
	generate_file_name((char *) PROCMAPDATAPATH,datarq.pid, file_name);
	sprintf(fileid_str, "%d", datarq.id);
	strcat(file_name,"_");
	strcat(file_name, fileid_str);
	fd = open(file_name, O_RDWR);
	//assert(fp);
	//fd = fileno(fp);
        if(fd < 0) {
	  fprintf(stderr,"file_name %s \n",file_name);	
	  assert(fd > -1);	
	}
	mmapobj->data_addr = (ULONG)mmap(0, datarq.bytes ,PROT_NV_RW, MAP_SHARED, fd, 0);

#ifdef _USEPIN
    //creates shared memory. if shared memory already created
    // then returns pointer
    init_pin();
    num_maps++;
    printf("Writing line %lu %lu %d\n", mmapobj->data_addr,mmapobj->data_addr+datarq.bytes,num_maps);
    Writeline((unsigned long)mmapobj->data_addr, (unsigned long)mmapobj->data_addr+datarq.bytes);
#endif


#else
	mmapobj->data_addr = (ULONG) map_nvram_state(&datarq);
#endif
	assert(mmapobj->data_addr);

	for (idx = 0; idx < mmapobj->numchunks; idx++) {

		nv_chunkobj = (chunkobj_s*) addr;
		//DEBUG_CHUNKOBJ(nv_chunkobj);

		if (check_modify_access(perm)) {

			chunkobj = nv_chunkobj;
			/*this wont work for restart. we need to initialize nvptr*/
#ifndef _DUMMY_TRANS
#ifdef _USE_SHADOWCOPY
			chunkobj->log_ptr =malloc(chunkobj->logptr_sz);
			chunkobj->nv_ptr = 0;
#ifdef _USE_UNDO_LOG
			assert(chunkobj->nv_ptr);
			record_chunks(chunkobj->nv_ptr,chunkobj);
#else
			assert(chunkobj->log_ptr);
			record_chunks(chunkobj->log_ptr,chunkobj);
#endif

#endif
#endif

		} else {
			chunkobj = nv_chunkobj;
		}
#ifdef _ENABLE_SWIZZLING
		//add the chunk objects to pointer tracing list
		//this is useful for tracing pointers during
		//recovery.
		//Beware: if the nv_ptr is null, assert will be false
		create_chunk_record(chunkobj->nv_ptr, chunkobj);
#endif

		char *ptr =  (char *) mmapobj->data_addr;
		//update the chunk pointer to new address
		chunkobj->nv_ptr = (void *) mmapobj->data_addr + chunkobj->offset;
		//fprintf(stdout,"chunk id %lu \n",(ULONG)chunkobj->chunkid);

		//add the chunk obj to mmap obj
		assert(add_chunkobj(mmapobj, chunkobj) == 0);

		DEBUG_CHUNKOBJ(chunkobj);
		//increment by chunk metadata size
		addr += (ULONG) sizeof(chunkobj_s);
		mmapobj->mmap_offset += (ULONG) sizeof(chunkobj_s);
	}
	DEBUG_T("Chunks. in MAP: %u: %u %lu\n",mmapobj->vma_id, idx,  mmapobj->data_addr);
	return 0;
}

/*Every NValloc call creates a mmap and each mmap
 is added to process object list*/
mmapobj_s* add_mmapobj_to_proc(proc_s *proc_obj, rqst_s *rqst, ULONG offset,
		ULONG data_addr) {

	mmapobj_s *mmapobj = NULL;

	mmapobj = create_mmapobj(rqst, offset, proc_obj, data_addr);
	assert(mmapobj);
	/*add mmap to process object*/
	add_mmapobj(mmapobj, proc_obj);
	proc_obj->pesist_mmaps++;

	return mmapobj;
}

static chunkobj_s* add_chunk_to_mmapobj(mmapobj_s *mmapobj, proc_s *proc_obj,
		rqst_s *rqst) {

	chunkobj_s *chunkobj = NULL;
	chunkobj = create_chunkobj(rqst, mmapobj);
	assert(chunkobj);

	/*add mmap to process object*/
	add_chunkobj(mmapobj, chunkobj);
	return chunkobj;
}

/*This method needs to be called only when process object
 * is created first time, and there is no persistent state
 */
int intialize_proc_obj(UINT pid, proc_s *proc_obj){

	proc_obj->pid = pid;
	proc_obj->size = 0;
	proc_obj->pesist_mmaps = 0;
	proc_obj->num_mmaps = 0;
	proc_obj->start_addr = 0;
	proc_obj->offset = 0;
	proc_obj->meta_offset = sizeof(proc_s);
	//proc_obj->data_map_size += rqst->bytes;
}

/*Idea is to have a seperate process map file for each process
 But fine for now as using browser */
static proc_s * create_or_load_proc_obj(int pid) {

	proc_s *proc_obj = NULL;
	size_t bytes = sizeof(proc_s);
	char file_name[256];
	int fd = -1;

	/*proc_obj = (proc_s *) mmap(0, PROC_METADAT_SZ, PROT_NV_RW,
			MAP_ANONYMOUS | MAP_PRIVATE, proc_map, 0);*/
	bzero(file_name, 256);
	generate_file_name((char *)PROCMETADATA_PATH, pid, file_name);

	/*we check if a map already exists*/
	fd = check_existing_map_file(file_name);
	if(fd < 0) {
		fd = setup_map_file(file_name, PROC_METADAT_SZ);
		assert(fd > -1);
		proc_obj = (proc_s *) mmap(0, PROC_METADAT_SZ, PROT_NV_RW,
				MAP_SHARED, fd, 0);
		assert(proc_obj != MAP_FAILED);
		memset ((void *)proc_obj,0,bytes)
		;
		intialize_proc_obj(pid,proc_obj);

	}else{
		proc_obj = (proc_s *) mmap(0, PROC_METADAT_SZ, PROT_NV_RW,
				MAP_SHARED, fd, 0);
		assert(proc_obj != MAP_FAILED);
	}
	proc_map_start = (ULONG) proc_obj;
	return proc_obj;
}

/*Func resposible for locating a process object given
 process id. The idea is that once we have process object
 we can get the remaining mmaps allocated process object
 YET TO COMPLETE */
static proc_s *find_proc_obj(int proc_id) {

	proc_s *proc_obj = NULL;

	if (enable_optimze) {
		if (proc_id && ((UINT) proc_id == prev_proc_id)) {

			if (prev_proc_obj) {
				return prev_proc_obj;
			}
		}
	}

	if (!proc_list_init) {
		proc_tree = rbtree_create();
		proc_list_init = 1;
		return NULL;
	}
	proc_obj = (proc_s *) rbtree_lookup(proc_tree, (void *) proc_id, IntComp);

	if (enable_optimze) {
		prev_proc_obj = proc_obj;
	}

	return proc_obj;
}



/*add process to the list of processes*/
static int add_proc_obj(proc_s *proc_obj) {

	if (!proc_obj)
		return 1;

	//if proecess list is not initialized,
	//then intialize it
	if (!proc_list_init) {
		proc_tree = rbtree_create();
		proc_list_init = 1;
	}
	rbtree_insert(proc_tree, (void *) proc_obj->pid, proc_obj, IntComp);

	/*we flush the proc obj*/
	if (useCacheFlush) {
		flush_cache(proc_obj, sizeof(proc_s));
	}
	return 0;
}

/*Function to find the process.
 @ pid: process identifier
 */
proc_s* find_process(int pid) {

	proc_s* t_proc_obj = NULL;

	if (!proc_list_init) {
		return NULL;
	}
	t_proc_obj = (proc_s*) rbtree_lookup(proc_tree, (void *) pid, IntComp);
	if(t_proc_obj)
		DEBUG("find_mmapobj found t_mmapobj %u \n", t_proc_obj->pid);
	return t_proc_obj;
}

//Return the starting address of process
ULONG get_proc_strtaddress(rqst_s *rqst) {

	proc_s *proc_obj = NULL;
	int pid = -1;
	uintptr_t uptrmap;

	pid = rqst->pid;
	proc_obj = find_proc_obj(pid);
	if (!proc_obj) {
		fprintf(stdout, "could not find the process. Check the pid %d\n", pid);
		return 0;
	} else {
		//found the start address
		uptrmap = (uintptr_t) proc_obj->start_addr;
		return proc_obj->start_addr;
	}
	return 0;
}

/*Initializes various process flags
 */
int nv_initialize(UINT pid) {

	proc_s	*procobj=NULL;

#ifdef _USE_CACHEFLUSH
	useCacheFlush=1;
#else
	useCacheFlush = 0;
#endif

#ifdef _NVRAM_OPTIMIZE
	enable_optimze=1;
	use_map_cache=1;
	next_mmap_entry = 0;
#else
	enable_optimze = 0;
	use_map_cache = 0;
#endif

#ifdef _USE_SHADOWCOPY
	enable_shadow=1;
#else
	enable_shadow = 0;
#endif

#ifdef _USE_UNDO_LOG
	enable_undolog=1;
#else
	enable_undolog = 0;
#endif

#ifdef _USE_TRANSACTION
#ifndef _DUMMY_TRANS
	enable_trans=1;
#else
	enable_trans=0;
#endif
#else
	enable_trans = 0;
#endif

#ifdef _NVSTATS
	enable_stats=1;
#else
	enable_stats = 0;
#endif

#ifdef ENABLE_LOGGING
	if (enable_trans) {
		assert(!initialize_logmgr(pid, 1));
	}
#endif

	//procobj = load_process(pid, 0);
	//if (!procobj) {
		//assert(procobj);
	//}	

}


//creates new  process structure
void* create_new_process(UINT pid) {

	proc_s *proc_obj = NULL;
	char *var = NULL;

	/*Before we create a new process we initialize various flags*/
	nv_initialize(pid);
	proc_obj = create_or_load_proc_obj(pid);
	assert(proc_obj);
	add_proc_obj(proc_obj);
	DEBUG_T("creating new proc %d\n", proc_obj->pid);
	return (void *) proc_obj;
}

/*Function to copy mmapobjs */
static int mmapobj_copy(mmapobj_s *dest, mmapobj_s *src) {

	if (!dest) {
		printf("mmapobj_copy:dest is null \n");
		goto null_err;
	}

	if (!src) {
		printf("mmapobj_copy:src is null \n");
		goto null_err;
	}

	//TODO: this is not a great way to copy structures
	dest->vma_id = src->vma_id;
	dest->length = src->length;
	dest->proc_id = src->proc_id;
	dest->offset = src->offset;
	//DEBUG_MMAPOBJ(dest);

	return 0;
	null_err: return -1;
}

/*gives the offset from start address
 * @params: start_addr start address of process
 * @params: curr_addr  curr_addr address of process
 * @return: offset
 */
ULONG findoffset(UINT proc_id, ULONG curr_addr) {

	proc_s *proc_obj = NULL;
	ULONG diff = 0;

	proc_obj = find_proc_obj(proc_id);
	if (proc_obj) {
		diff = curr_addr - (ULONG) proc_obj->start_addr;
		return diff;
	}
	return 0;
}

/* update offset for a mmapobj relative to start address
 * @params: proc_id
 * @params: vma_id
 * @params: offset
 * @return: 0 if success, -1 on failure
 */
int nv_record_chunk(rqst_s *rqst, ULONG addr) {

	proc_s *proc_obj;
	mmapobj_s *mmapobj = NULL;
	chunkobj_s *chunk = NULL;
	long vma_id = 0;
	UINT proc_id = 0, offset = 0;
	ULONG start_addr = 0;
	rqst_s lcl_rqst;

	//fprintf(stdout,"******************RECORD CHUNK1*****\n");
	assert(rqst);
	/*check if the process is already in memory*/
	proc_id = rqst->pid;
	proc_obj = find_proc_obj(proc_id);



	/*Try loading process from persistent memory*/
	if (!proc_obj) {

		//proc_obj = load_process(proc_id, 0 /*permission*/);
		/*if we cannot load a process, then create a new one*/
		if (!proc_obj) {
			proc_obj = (proc_s *) create_new_process(proc_id);
		} else {
			if (enable_trans) {
#ifdef ENABLE_LOGGING
				assert(!initialize_logmgr(proc_obj->pid, 0));
#endif
			}
		}
	}
	assert(proc_obj);

	//fprintf(stdout,"******************RECORD CHUNK2*****\n");
	if (use_map_cache) {
		vma_id = locate_mmapobj_node((void *) addr, rqst, &start_addr,
				&mmapobj);
	} else {
		vma_id = locate_mmapobj_node((void *) addr, rqst, &start_addr, NULL);
	}
	assert(vma_id);
	assert(start_addr);

	/*first we try to locate an exisiting obhect*/
	/*add a new mmapobj and update to process*/
	if (!mmapobj) {
		mmapobj = find_mmapobj(vma_id, proc_obj);
	}
	lcl_rqst.pid = rqst->pid;
	lcl_rqst.id = vma_id;

	if (!mmapobj) {
		lcl_rqst.bytes = get_vma_size(vma_id);
		mmapobj = add_mmapobj_to_proc(proc_obj, &lcl_rqst, offset, start_addr);
		assert(mmapobj);
	} else {
		lcl_rqst.bytes = mmapobj->length;
	}
	assert(lcl_rqst.bytes);

	/*find the mmapobj using vma id
	 if application has supplied request id, neglect*/
	if (rqst->var_name) {

		lcl_rqst.id = gen_id_from_str(rqst->var_name);
		DEBUG("generated chunkid %u from "
				"variable %s\n",lcl_rqst.id, rqst->var_name);
	} else {
		lcl_rqst.id = rqst->id;
		fprintf(stdout,"using id for alloc %u \n", lcl_rqst.id);
	}
	assert(lcl_rqst.id);

	//fprintf(stdout,"chunkid %u \n", lcl_rqst.id);

	//318730 for 100000 keys
	/*check if chunk already exists
	 This is an ambiguos situation
	 we do not know if programmar made an error
	 in naming or intentionally same chunk id
	 is used*/

	offset = addr - start_addr;
	prev_proc_id = rqst->pid;
	lcl_rqst.pid = rqst->pid;
	lcl_rqst.nv_ptr = (void *) addr;
	lcl_rqst.log_ptr = rqst->log_ptr;
	lcl_rqst.bytes = rqst->bytes;
	lcl_rqst.offset = offset;
	lcl_rqst.logptr_sz = rqst->logptr_sz;
	lcl_rqst.no_dram_flg = rqst->no_dram_flg;

#if 0
	chunk = find_chunk(mmapobj, lcl_rqst.id);
	if(chunk) {

#ifdef _NVDEBUG
		fprintf(stdout,"Chunk already exists %s id %u\n",
				rqst->var_name, lcl_rqst.id);
#endif
		//FIXME: What about garbage collecting old pointers
		//Garbage collection must be done
		update_chunkobj(&lcl_rqst, mmapobj,chunk);
	} else
#endif
#if 1
	{
		chunk = add_chunk_to_mmapobj(mmapobj, proc_obj, &lcl_rqst);
#ifdef _NVSTATS
		chunk->chunk_cnt = mmapobj->numchunks;
#endif
		mmapobj->numchunks++;
		DEBUG_CHUNKOBJ(chunk);
		DEBUG("adding chunk %s id: %d of size %u:"
				"to vma_id %u\n",
				rqst->var_name,
				lcl_rqst.id,
				lcl_rqst.bytes,vma_id);
	}
#endif

#ifndef _DUMMY_TRANS
#ifdef _USE_TRANSACTION

	//if (enable_shadow && enable_trans) {
	if (enable_trans) {
		if (enable_undolog) {
			assert(chunk->nv_ptr);
			record_chunks(chunk->nv_ptr, chunk);
		} else {
			assert(chunk->log_ptr);
			record_chunks(chunk->log_ptr, chunk);
		}
	}
#endif
#endif

	record_chunks(chunk->nv_ptr, chunk);


	//fprintf(stdout,"******************END RECORD CHUNK*****\n");
	//gt_spinlock_init(&chunk->chunk_lock);
	//add_record(chunk);
	return SUCCESS;
}

/*if not process with such ID is created then
 we return 0, else number of mapped blocks */
int get_proc_num_maps(int pid) {

	proc_s *proc_obj = NULL;
	proc_obj = find_proc_obj(pid);
	if (!proc_obj) {
		fprintf(stdout, "process not created \n");
		return 0;
	} else {
		//also update the number of mmapobj blocks
		return proc_obj->pesist_mmaps;
	}
	return 0;
}

int iterate_chunk_obj(rbtree_node n) {

	chunkobj_s *chunkobj = NULL;

	if (n == NULL) {
		return 0;
	}

	if (n->right != NULL) {
		iterate_chunk_obj(n->right);
	}

	chunkobj = (chunkobj_s *) n->value;
	if (chunkobj) {
		fprintf(stdout, "chunkobj chunkid: %d addr %lu\n", chunkobj->chunkid,
				(unsigned long) chunkobj);
		return 0;
	}

	if (n->left != NULL) {
		iterate_chunk_obj(n->left);
	}
	return 0;
}

int iterate_mmap_obj(rbtree_node n) {

	mmapobj_s *t_mmapobj = NULL;

	if (n == NULL) {
		return 0;
	}
	if (n->right != NULL) {
		iterate_mmap_obj(n->right);
	}
	t_mmapobj = (mmapobj_s *) n->value;
	if (t_mmapobj) {
		if (t_mmapobj->chunkobj_tree) {
			iterate_chunk_obj(t_mmapobj->chunkobj_tree->root);
		}
	}
	if (n->left != NULL) {
		iterate_mmap_obj(n->left);
	}
	return 0;
}

proc_s* load_process(int pid, int perm) {

	proc_s *proc_obj = NULL;
	proc_s *nv_proc_obj = NULL;
	mmapobj_s *mmapobj, *nv_mmapobj;
	size_t bytes = 0;
	int fd = -1;
	void *map;
	int idx = 0;
	ULONG addr = 0;
	char file_name[256];
	int wrt_perm = 0;

	//fprintf(stdout,"******************LOAD PROCESS*****\n");
	nv_initialize(pid);

	/*bzero(file_name, 256);
	bytes = sizeof(proc_s);
	generate_file_name((char *) PROCMETADATA_PATH, pid, file_name);
	fd = open(file_name, O_RDWR);
	if (fd == -1) {
		file_error(file_name);
		return NULL;
	}
	map = (proc_s *) mmap(0, PROC_METADAT_SZ,
			PROT_NV_RW, MAP_SHARED, fd, 0);

	nv_proc_obj = (proc_s *) map;
	if (nv_proc_obj == MAP_FAILED) {
		close(fd);
		DEBUG("Error mmapobjping the file %s\n", file_name);
		return NULL;
	}*/
	nv_proc_obj = (proc_s *)create_or_load_proc_obj(pid);

	/*Check if process has write permission to modify
	 *metadata
	 */
	wrt_perm = check_modify_access(perm);

	/*Start reading the mmapobjs
	 *add the process to proc_obj tree
	 *add initialize the mmapobj list if not
	 */
	addr = (ULONG) nv_proc_obj;
	addr = addr + sizeof(proc_s);

	DEBUG("WRITE permission %d\n", wrt_perm);

	if (wrt_perm) {
		proc_obj = nv_proc_obj;
	} else {
		proc_obj = (proc_s *) malloc(sizeof(proc_s));
		assert(proc_obj);
		proc_obj->pid = nv_proc_obj->pid;
		proc_obj->size = nv_proc_obj->size;
		proc_obj->pesist_mmaps = nv_proc_obj->pesist_mmaps;
		proc_obj->num_mmaps = nv_proc_obj->num_mmaps;
		proc_obj->start_addr = 0;
		if (enable_trans) {
			proc_obj->haslog = nv_proc_obj->haslog;
		}
	}
	proc_obj->mmapobj_initialized = 0;
	proc_obj->mmapobj_tree = 0;
	add_proc_obj(proc_obj);

	if (!proc_obj->mmapobj_initialized) {
		initialize_mmapobj_tree(proc_obj);
	}

	DEBUG_PROCOBJ(proc_obj);

	//Read all the mmapobjs
	for (idx = 0; idx < proc_obj->pesist_mmaps; idx++) {
		nv_mmapobj = (mmapobj_s*) addr;
		DEBUG_MMAPOBJ(nv_mmapobj);
		if (wrt_perm) {
			mmapobj = nv_mmapobj;
		} else {
			mmapobj = (mmapobj_s *) malloc(sizeof(mmapobj_s));
			copy_mmapobj(mmapobj, nv_mmapobj);
		}
		add_mmapobj(mmapobj, proc_obj);
		DEBUG_MMAPOBJ(nv_mmapobj);

		if (restore_chunk_objs(mmapobj, perm)) {
			fprintf(stdout, "failed restoration\n");
			goto error;
		}

#ifdef ENABLE_CHECKPOINT	
		if(mmapobj)
			record_vmas(mmapobj->vma_id, mmapobj->length);
		//record_vma_ghash(mmapobj->vma_id, mmapobj->length);
#endif
		addr = addr + sizeof(mmapobj_s);
		DEBUG_MMAPOBJ_T(mmapobj);
	}
	//fprintf(stdout,"******************END LOAD PROCESS*****\n");
	return proc_obj;
	error: return NULL;
}

//This function just maps the address space corresponding
//to process.
void* map_nvram_state(rqst_s *rqst) {

	void *nvmap = NULL;
	int fd = -1;
	nvarg_s a;

	//DEBUG_T("nvarg.proc_id %d %d %d\n",rqst->pid, rqst->bytes, rqst->id);
	//printf("nvarg.proc_id %d %d %d\n",rqst->pid, rqst->bytes, rqst->id);

	a.proc_id = rqst->pid;
	assert(a.proc_id);
	a.vma_id = rqst->id;
	a.pflags = 1;
	a.noPersist = 0;

#ifdef _USE_FAKE_NVMAP
	char file_name[MAX_FPATH_SZ];
	char fileid_str[64];

	bzero(file_name,256);
	generate_file_name((char *) PROCMAPMETADATA_PATH,rqst->pid, file_name);
	sprintf(fileid_str, "%d", rqst->id);
	strcat(file_name,"_");
	strcat(file_name, fileid_str);
	fd = setup_map_file(file_name, rqst->bytes);
	if (fd == -1) {
		perror("file open error\n");
		assert(fd > -1);
	}
	nvmap = mmap_wrap(0, rqst->bytes,
			PROT_NV_RW, MAP_SHARED , fd, 0, &a);
#else
	a.fd = -1;
	nvmap = mmap_wrap(0, rqst->bytes, PROT_NV_RW, PROT_ANON_PRIV, fd, 0, &a);
#endif
	assert(nvmap);
	if (nvmap == MAP_FAILED) {
		close(fd);
		goto error;
	}

	/*chunkobj_s *nv_chunkobj;
	if(a.vma_id != 2781) {
	    a.vma_id  = 2781;
	    nv_chunkobj = (chunkobj_s*) mmap_wrap(0, rqst->bytes, PROT_NV_RW, PROT_ANON_PRIV, fd, 0, &a);
		if(nv_chunkobj)
		DEBUG_CHUNKOBJ_T(nv_chunkobj);
	}*/


	//fprintf(stdout,"******************LOAD PROCESS END*****\n");
	return nvmap;
	error: return NULL;
}

//////////////////FUNCTIONS USED FOR RECOVERY/////////////////////////////////////////////////
int init_pntrlst_tree() {

	pntrlst_tree = rbtree_create();
	pntrlst_tree_init = 1;
	return 0;
}





#ifdef _ENABLE_SWIZZLING

int CompPtrLstRange(node key_node, void* a, void* b) {

	chunkobj_s *chunksrc =
			(chunkobj_s*) key_node->value;

	ULONG a_start_addr = (ULONG) a;
	ULONG b_start_addr = (ULONG) b;
	ULONG chunk_end_addr = chunksrc->length + b_start_addr;


	if ((a_start_addr >= b_start_addr)
			&& (chunk_end_addr > a_start_addr)) {
		return (0);
	}
	if (a_start_addr > b_start_addr)
		return (1);
	if (a_start_addr < b_start_addr)
		return (-1);
	return (0);
}

chunkobj_s *find_from_ptrlst(void *addr) {

	chunkobj_s *chunk;

	if (!addr)
		return NULL;

	chunk = (chunkobj_s *)rbtree_lookup(pntrlst_tree,(void *)addr, CompPtrLstRange);
	if(chunk){
		return chunk;
	}
	return NULL;
}

int create_chunk_record(void* addr, chunkobj_s *chunksrc) {

	chunksrc->old_nv_ptr = chunksrc->nv_ptr;
	if(!pntrlst_tree_init)
		init_pntrlst_tree();
	rbtree_insert(pntrlst_tree, (void*)chunksrc->old_nv_ptr, chunksrc, IntComp);
	return 0;
}

//FIXME: Error in getting chunk just using
//chunk id without even using process id
void *load_valid_addr(void **ptr) {

	chunkobj_s *oldchunk, *newchunk;
	ULONG offset=0;
	void *nv_ptr= NULL;

	if(ptr == NULL)
		return NULL;

	if(NULL == *ptr)
		return NULL;

	oldchunk = find_from_ptrlst(*ptr);
	if(!oldchunk) {
		//could not find an equivalent pointer
		return NULL;
	}
	//Ok we found a chunk corresponding to
	//this pointed. Now lets find, in what
	//offset of chunk was this pointed located
	offset = (ULONG)*ptr - (ULONG)oldchunk->old_nv_ptr;

	//lets find the current NV_PTR for this chunk
	rqst_s rqst;
	rqst.pid = oldchunk->pid;
	rqst.id = oldchunk->chunkid;
	rqst.var_name = NULL;

	nv_ptr = nv_map_read(&rqst, NULL /*always NULL*/);
	//All old states must have been loaded
	assert(nv_ptr);
	*ptr = nv_ptr + offset;
	DEBUG_T("oldchunk->chunkid %d\n", oldchunk->chunkid);
	//fprintf(stdout,"loaded start addr %lu, *ptr %lu, offset %lu\n", (unsigned long)nv_ptr, *ptr,offset);
	return *ptr;
}

#if 0
//delete the comments later
chunkobj_s *find_from_ptrlst(void *addr) {

	chunkobj_s *chunk;

	if (!addr)
		return NULL;

	chunk = (chunkobj_s *)rbtree_lookup(pntrlst_tree,(void *)addr, CompPtrLstRange);
	if(chunk){
		return chunk;
	}

	/*size_t bytes = 0;
	std::map<void*, chunkobj_s *>::iterator ptritr;
	unsigned long ptr = (unsigned long) addr;
	unsigned long start, end;
	for (ptritr = pntr_lst.begin(); ptritr != pntr_lst.end(); ++ptritr) {
		chunk = (chunkobj_s *) (*ptritr).second;
		bytes = chunk->length;
		start = (ULONG) (*ptritr).first;
		end = start + bytes;
		if (ptr >= start && ptr < end) {
			return chunk;
		}
	}*/
	return NULL;
}


int create_chunk_record(void* addr, chunkobj_s *chunksrc) {

	chunksrc->old_nv_ptr = chunksrc->nv_ptr;
	if(!pntrlst_tree_init)
		init_pntrlst_tree();
	rbtree_insert(pntrlst_tree, (void*)chunksrc->old_nv_ptr, chunksrc, IntComp);

	/*chunkobj_s *chunkcpy = (chunkobj_s *)malloc(sizeof(chunkobj_s));
	copy_chunkoj(chunkcpy,chunksrc);
	assert(chunksrc->nv_ptr);
	//copy chunk, does not copy nvptr, as might be stale
	//in some case. so we explicity check for null and
	//if not copy
	chunkcpy->old_nv_ptr = chunksrc->nv_ptr;
	pntr_lst[addr] = chunkcpy;
	DEBUG_T("creating records %lu, chunk id %u\n", addr, chunkcpy->chunkid);*/

	return 0;
}

int delete_chunk_record() {

	std::map <void*, chunkobj_s *>::iterator ptritr;

	for( ptritr= pntr_lst.begin(); ptritr!=pntr_lst.end(); ++ptritr) {
		chunkobj_s *chunk = (chunkobj_s *)(*ptritr).second;
		if(chunk)
			free(chunk);
	}
	return 0;
}
#endif

#endif


void* nv_map_read(rqst_s *rqst, void* map) {

	ULONG offset = 0;
	ULONG addr_l = 0;
	int process_id = 1;
	proc_s *proc_obj = NULL;
	unsigned int chunk_id;
	mmapobj_s *mmapobj_ptr = NULL;
	void *map_read = NULL;
	rbtree_t *tree_ptr;
	chunkobj_s *chunkobj;
	int perm = rqst->access;
	char *del;

#ifdef _USE_DISKMAP
	char out_fname[256];
	FILE *fp = NULL;
	int fd = -1;
	struct stat statbuf;
#endif


	process_id = rqst->pid;
	//Check if all the process objects are still in memory and we are not reading
	//process for the first time
	proc_obj = find_process(rqst->pid);
	if (!proc_obj) {
		/*looks like we are reading persistent structures
		 and the process is not avaialable in memory
		 FIXME: this just addressies one process,
		 since map_read field is global*/
		proc_obj = load_process(process_id, perm);
		if (!proc_obj) {
			assert(proc_obj);
			goto error;
		}
	}
	if (rqst->var_name) {
		chunk_id = gen_id_from_str(rqst->var_name);
	} else {
		chunk_id = rqst->id;
	}

	mmapobj_ptr = find_mmapobj_from_chunkid(chunk_id, proc_obj);
	if (!mmapobj_ptr) {
		fprintf(stdout,"finding chunk %s withid %u failed \n", rqst->var_name,chunk_id);
		assert(0);
		goto error;
	}

	//Get the the start address and then end address of mmapobj
	/*Every malloc call will lead to a mmapobj creation*/
	tree_ptr = mmapobj_ptr->chunkobj_tree;
	chunkobj = (chunkobj_s *) rbtree_lookup(tree_ptr, (void*) chunk_id,
			IntComp);

	assert(chunkobj);
	assert(chunkobj->nv_ptr);
	assert(chunkobj->length);

#ifdef _USE_SHADOWCOPY
	chunkobj->log_ptr = malloc(chunkobj->length);
	assert(chunkobj->log_ptr);
	memcpy(chunkobj->log_ptr, chunkobj->nv_ptr, chunkobj->length);
	rqst->log_ptr= chunkobj->log_ptr;
#endif
	rqst->nv_ptr = chunkobj->nv_ptr;
	rqst->bytes = chunkobj->length;

#ifdef _NVSTATS
	if(enable_stats) {
		add_stats_chunk_read(rqst->pid, chunkobj->length);
		print_stats(rqst->pid);
	}
#endif

#ifdef VALIDATE_CHKSM
	assert(compare_checksum(chunkobj->nv_ptr, 
			chunkobj->length,
			chunkobj->checksum) == 0);
#endif

	//fprintf(stdout, "nvmmap nvptr: %lu \n",(ULONG)chunkobj->nv_ptr);
	//fprintf(stdout, "chunk_id: %lu \n",(ULONG)chunk_id);

	return (void *) rqst->nv_ptr;
	error: rqst->nv_ptr = NULL;
	rqst->log_ptr = NULL;
	return NULL;
}

int nv_munmap(void *addr, size_t size) {

	int ret_val = 0;

	if (!addr) {
		perror("null address \n");
		return -1;
	}
	ret_val = munmap(addr, size);
	return ret_val;
}

void *create_map_tree() {

	if (!map_tree)
		map_tree = rbtree_create();

	if (!map_tree) {
		perror("RB tree creation failed \n");
		exit(-1);
	}
	return map_tree;
}

int insert_mmapobj_node(ULONG val, size_t size, int id, int proc_id) {

	struct mmapobj_nodes *mmapobj_struct;
	mmapobj_struct = (struct mmapobj_nodes*) malloc(
			sizeof(struct mmapobj_nodes));
	mmapobj_struct->start_addr = val;
	mmapobj_struct->end_addr = val + size;
	mmapobj_struct->map_id = id;
	mmapobj_struct->proc_id = proc_id;

	if (!map_tree)
		create_map_tree();

	rbtree_insert(map_tree, (void*) val, mmapobj_struct, IntComp);
	return 0;
}

UINT locate_mmapobj_node(void *addr, rqst_s *rqst, ULONG *map_strt_addr,
		mmapobj_s **mmap_obj) {

	struct mmapobj_nodes *mmapobj_struct = NULL;
	ULONG addr_long, strt_addr;
	UINT mapid;

	if (use_map_cache) {
		mmapobj_s *tmpobj;
		tmpobj = get_frm_mmap_cache(addr);
		if (tmpobj) {
			*map_strt_addr = tmpobj->data_addr;
			assert(mmap_obj);
			*mmap_obj = tmpobj;
			return tmpobj->vma_id;
		}
	}

	if (mmap_obj)
		*mmap_obj = NULL;

	addr_long = (ULONG) addr;
	mmapobj_struct = (struct mmapobj_nodes *) rbtree_lookup(map_tree,
			(void *) addr_long, CompRange);
	if (mmapobj_struct) {
		mapid = mmapobj_struct->map_id;
		strt_addr = mmapobj_struct->start_addr;
		*map_strt_addr = strt_addr;
		return mapid;
	}
	if (enable_debug) {
		fprintf(stdout, "query failed pid:%d %u addr: %lu\n", rqst->pid,
				rqst->id, addr_long);
	}
	return 0;
}

size_t total_size = 0;
void* _mmap(void *addr, size_t size, int mode, int prot, int fd, int offset,
		nvarg_s *a) {

	void *ret = NULL;
	ULONG addr_long = 0;
	proc_s *proc_obj = NULL;

	assert(a);
	assert(a->proc_id);

	proc_obj = find_proc_obj(a->proc_id);
	if (!proc_obj) {
		proc_obj = create_or_load_proc_obj(a->proc_id);
		assert(proc_obj);
	}

	a->fd = -1;
	a->vma_id = proc_obj->num_mmaps+1;
	proc_obj->num_mmaps++;
	a->pflags = 1;
	a->noPersist = 0;
	total_size += size;

#ifdef _USE_FAKE_NVMAP
	char file_name[MAX_FPATH_SZ];
	char fileid_str[64];
	//fprintf(stdout,"%zu \n",size);
	if(size >= 33550336){
	//if(1){
		bzero(file_name,256);
		generate_file_name((char *) PROCMAPDATAPATH,a->proc_id, file_name);
		sprintf(fileid_str, "%d", a->vma_id);
		strcat(file_name,"_");
		strcat(file_name, fileid_str);
		fd = setup_map_file(file_name, size);
		if (fd == -1) {
			perror("file open error\n");
			assert(fd > -1);
		}
		ret = mmap_wrap(addr, size,PROT_NV_RW, MAP_SHARED , fd, 0, a);
	}
	else{
         ret = mmap_wrap(addr, size, mode, prot, fd, offset, a);
		 //ret = malloc(size);
	}
#else
    //if(size >= 33550336){
		ret = mmap_wrap(addr, size, mode, prot, fd, offset, a);
	//}else{
	//	ret = mmap (addr, size, mode, prot, fd, offset);
	//}
#endif
	assert(ret);

#ifdef _NVSTATS
	if(enable_stats) {
		incr_proc_mmaps();
		add_stats_mmap(a->proc_id, size);
	}
#endif
	addr_long = (ULONG) ret;
	insert_mmapobj_node(addr_long, size, a->vma_id, a->proc_id);
	record_vmas(a->vma_id, size);
	return ret;
}

/*Called by nv_commit, or undo log
 * takes rqst as input, with mandatatory
 * process id, and varname/chunkid as param
 */
chunkobj_s * get_chunk(rqst_s *rqst) {

	int process_id = -1;
	int ops = -1;
	proc_s *proc_obj = NULL;
	unsigned int chunkid = 0;
	mmapobj_s *mmapobj_ptr = NULL;

	if (!rqst)
		return NULL;

	//we don't need size if transaction,
	//we just commit the entire object
	if (enable_trans) {
		assert(rqst->bytes);
	}

	process_id = rqst->pid;
	assert(process_id);

	if (enable_trans) {
		ops = rqst->ops;
	}
	proc_obj = find_proc_obj(process_id);
	assert(proc_obj);
	/*find the mmapobj if application has
	 supplied request id, neglect
	 can just supply the transaction ID*/
	if (!rqst->id) {
		if (rqst->var_name)
			chunkid = gen_id_from_str(rqst->var_name);
		else
			printf("nv_commit:error generating vma id \n");
	} else {
		chunkid = rqst->id;
	}
	/*we always make assumption the
	 chunk id is always greater than 1*/
	assert(chunkid);
	//we verify if such a chunk exists
	mmapobj_ptr = find_mmapobj_from_chunkid(chunkid, proc_obj);
	assert(mmapobj_ptr);
	return find_chunk(mmapobj_ptr, chunkid);
}

int nv_commit(rqst_s *rqst) {

	chunkobj_s *chunk = NULL;
	void* dest = NULL;
	void *src = NULL;

	chunk = get_chunk(rqst);
	assert(chunk);
	assert(chunk->length);

#ifdef _USE_SHADOWCOPY
	src = chunk->log_ptr;
	dest =chunk->nv_ptr;
#else
	src = chunk->nv_ptr;
	dest = chunk->nv_ptr;
#endif
	assert(src);
	assert(dest);
	memcpy(dest, src, chunk->length);

#ifdef _USE_TRANSACTION
	chunk->dirty = 1;
#endif

#ifdef _NVDEBUG
	fprintf(stderr,"nv_commit: COMPLETE \n");
#endif
	return 0;
}

int app_exec_finish(int pid) {

#ifdef _USE_DISKMAP
	disk_flush(pid);
#endif

#ifdef _ASYNC_RMT_CHKPT
	send_lock_avbl_sig(SIGUSR2);
#endif

#ifdef _NVSTATS
	if(enable_stats) {
		print_stats(pid);
	}
#endif

#ifdef _USE_TRANSACTION
	print_trans_stats();
#endif

	return 0;
}

//#ifdef _NVRAM_OPTIMIZE
void add_to_mmap_cache(mmapobj_s *mmapobj) {

	mmap_strt_addr_cache[next_mmap_entry]= mmapobj->data_addr;
	mmap_size_cache[next_mmap_entry]= mmapobj->length;
	mmap_ref_cache[next_mmap_entry] = (ULONG)mmapobj;
	//fprintf(stdout,"startaddr %lu next_mmap_entry %lu\n",mmapobj->strt_addr, next_mmap_entry);
	next_mmap_entry++;
	next_mmap_entry = next_mmap_entry % NUM_MMAP_CACHE_CNT;
}

mmapobj_s* get_frm_mmap_cache(void *addr) {

	UINT idx =0;
	ULONG target = (ULONG)addr;

	for (idx =0; idx < NUM_MMAP_CACHE_CNT; idx++) {

		ULONG startaddr = mmap_strt_addr_cache[idx];

		if (startaddr <= target &&
				startaddr + mmap_size_cache[idx] > target)
		{
			return (mmapobj_s*) mmap_ref_cache[idx];
		} else {
			//fprintf(stdout,"miss \n");
		}
	}
	return NULL;
}

//#endif


#ifdef _USE_BASIC_MMAP
void* create_mmap_file(rqst_s *rqst) {

   void *ret = NULL;
   ULONG addr = 0;
   int fd = -1;
   char file_name[256];

   assert(rqst);
   assert(rqst->var_name);

   bzero(file_name,256);
   strcpy(file_name,BASEPATH);
   strcat(file_name, rqst->var_name);

	fd = setup_map_file(file_name, rqst->bytes);
	if (fd == -1) {
		perror("file open error\n");
		assert(fd > -1);
	}
	ret = mmap((void *)addr, rqst->bytes,PROT_NV_RW, MAP_SHARED, fd, 0);
	assert(ret);
	close(fd);

	return ret;
}

void* read_mmap_file( rqst_s* rqst, size_t *len){

   void *mmapfile = NULL;
   struct stat sb;
   char file_name[256];

   assert(rqst);
   assert(rqst->var_name);	

   bzero(file_name,256);
   strcpy(file_name,BASEPATH);
   strcat(file_name, rqst->var_name);


   /*we check if a map already exists*/
   int fd = check_existing_map_file(file_name);
   if(fd <= -1){ 	
   	   fprintf(stdout,"file_name %s \n", file_name);
	   assert(fd > -1);	
	}
	if (stat(file_name, &sb) == -1) {
        perror("stat");
        exit(EXIT_FAILURE);
    }
  *len = sb.st_size;
   mmapfile = (void *) mmap(0,sb.st_size,PROT_NV_RW, MAP_SHARED, fd, 0);
   assert(mmapfile != MAP_FAILED);

   close(fd);

   return mmapfile;	
}

void close_mmap_file( rqst_s* rqst){

   void *mmapfile = NULL;
   struct stat sb;
   size_t size=0;
   int fd = -1;
   char file_name[256];

   assert(rqst);
   assert(rqst->var_name);	
   assert(rqst->nv_ptr);	

   bzero(file_name,256);
   strcpy(file_name,BASEPATH);
   strcat(file_name, rqst->var_name);

   /*we check if a map already exists*/
   //fd = check_existing_map_file(file_name);
   //if (fd < 0 ) {
	//fprintf(stdout,"file_name %s \n",file_name);
	//assert(fd > -1);	
   //}	
   if (stat(file_name, &sb) == -1) {
      perror("stat");
      perror(file_name);
      exit(EXIT_FAILURE);
   }
   size= sb.st_size;
   nv_munmap(rqst->nv_ptr, size);
   //if(fd)
   //close(fd);		
}

#endif
int nvm_persist(void *addr, size_t len, int flags){
	//pmem_persist(addr,len, flags);
	flush_cache(addr,len);
}




#if 1

unsigned int spin_init =0, g_fault_count=0;
struct gt_spinlock_t spinlock_fault;
//extern struct gt_spinlock_t spin_lock;
unsigned long *fault_count;

/*******************FAULT HANDLERS*****************************************/

static void SIGABRT_handle(int sig, siginfo_t *si, void *unused)
{
	fprintf(stdout,"handling SIGABRT \n");
}


static void SIGTERM_handle(int sig, siginfo_t *si, void *unused)
{
	fprintf(stdout,"handling SIGTERM \n");
}


static void SIGILL_handle(int sig, siginfo_t *si, void *unused)
{
	fprintf(stdout,"handling SIGTERM \n");
}


static void fault_handler (int sig, siginfo_t *si, void *unused)
{
    disable_alloc_prot((void *)si->si_addr);
}

/*************************************************************************/


void enable_alloc_prot(void* ptr, size_t size){

	struct sigaction sa, sa1, sa2, sa3;



#if 0 //Disable this, caller takes care of handling faults
	std::map <void *, size_t>::iterator itr;
    memset (&sa, 0, sizeof (sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = fault_handler;
    sa.sa_flags   = SA_SIGINFO;
    if (sigaction(SIGSEGV, &sa, NULL) == -1)
        handle_error("sigaction"); 

    /*memset (&sa1, 0, sizeof (sa1));
    sigemptyset(&sa1.sa_mask);
    sa1.sa_handler = SIGABRT_handle;
    sa1.sa_flags   = SA_SIGINFO;
    if (sigaction(SIGABRT, &sa1, NULL) == -1)
        handle_error("sigaction"); 

    memset (&sa2, 0, sizeof (sa2));
    sigemptyset(&sa2.sa_mask);
    sa2.sa_handler = SIGTERM_handle;
    sa2.sa_flags   = SA_SIGINFO;
    if (sigaction(SIGTERM, &sa2, NULL) == -1)
        handle_error("sigaction"); 


    memset (&sa3, 0, sizeof (sa3));
    sigemptyset(&sa3.sa_mask);
    sa3.sa_handler = SIGILL_handle;
    sa3.sa_flags   = SA_SIGINFO;
    if (sigaction(SIGILL, &sa3, NULL) == -1)
        handle_error("sigaction"); */
#endif

	if(size < 4096) {
		return;
	}
	size = size - size%4096-1;
	//add_alloc_map(ptr, size);
	if (mprotect((void *)ptr,size, PROT_READ)==-1) {
          exit(-1);   
    }
	//alloc_prot_map[ptr] = 1;
	/*life_map[ptr]=1;
    for( itr= life_map.begin(); itr!=life_map.end(); ++itr){
   	    void *addr = (void *)(*itr).first;
		int lifecount = life_map[addr];
		if(lifecount > 100) {
			alloc_prot_map.erase(addr);
			allocmap.erase(addr);
			life_map.erase(addr);
		}	
	}*/	
}





/* function to add protection to all the chunks
 * allocated with NVM library
 */
void protect_all_chunks(){

    std::unordered_map<unsigned long, size_t>::iterator itr;

    for( itr= allocmap.begin(); itr!=allocmap.end(); ++itr){
   	    void *addr = (void *)(*itr).first;
		int isfault;

		if(!addr) continue;

		isfault = alloc_prot_map[addr];
		isfault = 0;
		if(isfault == 0){
			if (mprotect((void *)addr,(size_t)(*itr).second,PROT_READ)==-1) {
   		      //exit(-1);
			}
			alloc_prot_map[addr]=1;
		}
	}
}


/*Function to disable protection of an address chunk
 *
 */
size_t  disable_alloc_prot(void *addr){
	size_t size = 0;
	unsigned long faddr; 
	unsigned long laddr = 0;
	unsigned long off = 0;
	size_t protect_len =0;


	//disable_malloc_hook();
	gt_spin_lock(&spinlock_fault);

	//alloc_prot_map[addr] = 0;
	size = get_alloc_size(addr, &faddr);
	if(faddr && size){
		fault_stat[faddr]++;
		alloc_prot_map[faddr] = 0;
	}


	//assert(faddr);
	//if(faddr)alloc_prot_map[faddr]=0;
	/*if(faddr && size){
 		 laddr = faddr - ((unsigned long)faddr%4096);
		 //size = 4095;
		 alloc_prot_map[faddr] = 0;
		 //life_map[faddr] = 0;
		 //fault_stat[faddr]++;
		 //fprintf(stdout,"%lu %u\n",faddr, fault_stat[faddr]);
	}
    else*/{
    	 /* align to page boundaries */
		 laddr = addr - ((unsigned long)addr%4096);
		 size = 4095;
	}

	if (mprotect((void *)laddr,size, PROT_READ|PROT_WRITE)==-1) {
           //exit(-1);   
    }

	/*keep track of fault count*/
	g_fault_count++;

	/*debugging purpose, need to be removed*/
	if(g_fault_count%10000 == 0){
		fprintf(stderr,"g_fault_count %u %u\n",g_fault_count, allocmap.size());
		//print_stat();
	}

	//enable_malloc_hook();
	gt_spin_unlock(&spinlock_fault);
	//gt_spin_unlock(&spin_lock);
	return size;
}


void print_stat() {

    std::unordered_map <unsigned long, size_t>::iterator itr;
	unsigned int count = 0;
	size_t size = fault_stat.size();
	unsigned long addr=0;
	size_t chunksz=0;

    for( itr= fault_stat.begin(); itr!=fault_stat.end(); ++itr){
        size_t fcount = (size_t)(*itr).second;

		if(fcount > 100) {

			addr = (unsigned long)(*itr).first;
			chunksz = allocmap[addr];	
	        //fprintf(stderr,"chunk %u, fault_count %zu, "
			//				"num. chunks  %zu chunksz %u "
			//				"hot chunks %u\n",
			//				count, fcount, size, chunksz, count);
			count++;
		}
    }
	fprintf(stdout,"hot chunks %u\n", count);

	fprintf(stdout,"-----------------\n");
}


void  remove_map(void *addr){

	size_t size = 0;
	unsigned long faddr; 

	//return 1;
	//disable_malloc_hook();
	size = get_alloc_size(addr, &faddr);
	 //if(faddr && alloc_prot_map.find(faddr) != alloc_prot_map.end() &&  allocmap.find(faddr) != allocmap.end()){
	 if(faddr && allocmap.find(faddr) != allocmap.end()){
		 //fprintf(stdout,"removing element \n");
		 //allocmap.erase(faddr);
		 //allocmap.erase(addr);
		 alloc_prot_map.erase(faddr);	
	}
	//enable_malloc_hook();
}

#endif
