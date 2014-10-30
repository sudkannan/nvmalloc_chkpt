#include "checkpoint.h"
#include <iostream>
#include <string>
#include <map>
#include <signal.h>
#include <queue>
#include <list>
#include <algorithm>
#include <functional>
#include <pthread.h>
#include <sys/time.h>
#include "nv_def.h"
#include "util_func.h"
#include "hash_maps.h"
#include "checkpoint.h"


#ifdef _ARMCI_CHECKPOINT
#include "armci_checkpoint.h"
#endif

#ifdef _ARMCI_CHECKPOINT
#define VERSION_INFO_SZ 64
#endif

#define MIN_ASYNC_DATA_LEN 2*1024*1024
//#define NV_DEBUG

using namespace std;

std::queue<int> sig_queue;
std::list<int> chunk_fault_list;
cktptlock_s *g_chkptlock;

static sigset_t newmask;
static sigset_t term_sig_mask;
extern unsigned int prev_proc_id;
static void sig_remote_chkpt(int signo);

#ifdef _NVSTATS
struct timeval r_start,r_end;
struct timeval r_ckpt_start,r_ckpt_end;
s_rmtchkptstat rstat;
#endif

//extern std::unordered_map <void *, chunkobj_s *> chunkmap;
//extern std::unordered_map <void *, chunkobj_s *>::iterator chunk_itr;



#if 0
std::unordered_map <int, size_t> proc_vmas;
std::unordered_map <int, size_t> metadata_vmas;
std::unordered_map<int, size_t>::iterator vma_itr;
std::unordered_map <void *, chunkobj_s *> chunkmap;
std::unordered_map <void *, chunkobj_s *>::iterator chunk_itr;
std::unordered_map<int, chunkobj_s *> id_chunk_map;

void* get_chunk_from_map(void *addr) {

	size_t bytes = 0;
	std::map <int, chunkobj_s *>::iterator id_chunk_itr;
	unsigned long ptr = (unsigned long)addr;
    unsigned long start, end;
	
    for( chunk_itr= chunkmap.begin(); chunk_itr!=chunkmap.end(); ++chunk_itr){
        chunkobj_s * chunk = (chunkobj_s *)(*chunk_itr).second;
        bytes = chunk->length;
		start = (ULONG)(*chunk_itr).first;
		end = start + bytes;
#ifdef NV_DEBUG
		//fprintf(stderr,"fetching %ld start %ld end %ld\n",ptr, start, end);
#endif

	    /*if(end) {
    	    unsigned long off = 0;
        	off = 4096- (((unsigned long)end % 4096));
	        fprintf(stdout,"off %d \n", off);
			end = end + off;
	   }*/
		if( ptr >= start && ptr <= end) {
			return (void *)chunk;
		}
	}
	return NULL;
}

void* get_chunk_with_id(UINT chunkid){

	return (void *)id_chunk_map[chunkid];
}


int record_chunks(void* addr, chunkobj_s *chunk) {

	//fprintf(stdout,"recording addr %lu\n",addr);
	chunkmap[addr] = chunk;
	//id_chunk_map[chunk->chunkid] = chunk;
	return 0;
}

int get_chnk_cnt_frm_map() {

	return chunkmap.size();
}

//Assuming that chunkmap can get data in o(1)
//Memory address range will not work here
//If this method returns NULL, caller needs
//to check if addr is in allocated ranger using
//the o(n) get_chunk_from_map call
void *get_chunk_from_map_o1(void *addr) {

	chunkobj_s *chunk;

	assert(chunkmap.size());
	assert(addr);
	//fprintf(stdout,"fetching addr %lu\n",addr);
	chunk = ( chunkobj_s *)chunkmap[addr];
	return (void *)chunk;
}


int record_vmas(int vmaid, size_t size) {

    proc_vmas[vmaid] = size;
    return 0;
}

int record_metadata_vma(int vmaid, size_t size) {

    metadata_vmas[vmaid] = size;
    return 0;
}


size_t get_vma_size(int vmaid){

	/*vma_itr=proc_vmas.find(vmaid); 
	if(vma_itr == proc_vmas.end())
		return 0;*/
	return proc_vmas[vmaid];
}

#endif

#ifdef _USE_CHECKPOINT
int get_vma_dirty_pgcnt(int procid, int vmaid) {

	UINT numpages;
	struct nvmap_arg_struct a;
	size_t bytes =0;
	void *dirtypgbuff = NULL;
    UINT offset = 0;

	a.fd = -1;
    a.offset = offset;
    a.vma_id =vmaid;
    a.proc_id =procid;
    a.pflags = 1;
    a.noPersist = 0;

	bytes = INTERGER_BUFF * sizeof(unsigned int);
	dirtypgbuff =   malloc(bytes);
	dirtypgbuff = dirtypgbuff + PAGE_SIZE;
	numpages =syscall(__NR_copydirtpages, &a, dirtypgbuff);
    fprintf(stdout, "Get dirty pages %d \n",numpages);

	return numpages;
}

int copy_dirty_pages(int procid, int vmaid, void *buffer, int bytes)
{
    void *map;
    UINT offset = 0;
    struct nvmap_arg_struct a;

    a.fd = -1;
    a.offset = offset;
    a.vma_id =vmaid;
    a.proc_id =procid;
    a.pflags = 1;
    a.noPersist = 0;
    
    if(bytes){
#ifndef _USE_FAKE_NVMAP
        map= (void *) syscall(__NR_nv_mmap_pgoff, 0, bytes,
                     PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS , &a );
#else
		map= (void *) mmap(0, bytes,PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS,-1, 0);
#endif
        if (map == MAP_FAILED) {
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }
        memcpy(buffer, map, bytes);
    }
    return 0;
}


void *copy_header(void *buff, int procid, int storeid, 
		size_t bytes, int type){

	chkpthead_s ckptstruct;

	ckptstruct.pid = procid;
	ckptstruct.type = type;
    ckptstruct.storeid = storeid;
	ckptstruct.bytes = bytes;

	assert(buff);
	memcpy(buff, &ckptstruct, sizeof(chkpthead_s));

	return buff;
}


void* helper_update_sendbuff(void *sendbuff, 
						void *tmp,
						size_t bytes,
						size_t *sent_bytes,
						int procid,
						int type,
						int vmaid) {

	size_t header_sz =0;
	header_sz = sizeof(chkpthead_s);

	tmp = sendbuff;
	tmp = tmp + *sent_bytes;
	copy_header(tmp, procid, vmaid,bytes, type);
	tmp = tmp + header_sz;
	return tmp;
}



void print_header( chkpthead_s *header) {

	assert(header);

	fprintf(stdout, "header->pid %d \n",header->pid);
	fprintf(stdout, "header->type %d \n",header->type);
	fprintf(stdout, "header->storeid %d \n",header->storeid);
	fprintf(stdout, "header->bytes %d \n",header->bytes);

}


int add_nvram_data(chkpthead_s *header) {

	rqst_s rqst;
	void *ret =0;

    rqst.pid = header->pid;
	rqst.id = header->storeid;
	rqst.bytes = header->bytes;
    assert(rqst.id);
	ret = map_nvram_state(&rqst);

#ifdef _DEBUG
	fprintf(stdout,"finshed mapping nvram state \n");
#endif

	assert(ret);
	return 0;
}


int parse_data(void *buffer, size_t size) {


	size_t header_sz =0;
	void *tmpbuff = NULL;
	chkpthead_s header, tmpheader;
    size_t itr =0;

	assert(buffer);
	header_sz = sizeof(chkpthead_s);

	while (itr < size) {

		memcpy((void *)&tmpheader, buffer, header_sz);

#ifdef _DEBUG
		print_header(&tmpheader);
#endif

		add_nvram_data(&tmpheader);

		buffer = buffer + tmpheader.bytes + header_sz;

		itr = itr + tmpheader.bytes + header_sz;
	}

	return 0;
}



void* get_shm(int id, int flag) {

        key_t key;
        int shmid;
        void *shm;
    
        key = id + SHM_BASE;

        if ((shmid = shmget(key,SHMSZ, flag)) < 0) {
#ifdef _DEBUG
			fprintf(stdout, "getting shm %d \n", key);
#endif
			//perror("shmget");
            return NULL;
        }

        if ((shm = shmat(shmid, (void *)0, 0)) == (void *)-1) {
    	    //perror("shmat");
            return NULL;
        }

        return shm;
}
 

cktptlock_s *create_shm_lock(int id) {

	void *shm;
	cktptlock_s *lock;
	int flag = 0;

	flag = 0666 | IPC_CREAT;

	shm = get_shm(id,flag);
	assert(shm);

	lock = (cktptlock_s *)shm;

	lock->dirty = 0;	

	return lock;
}

cktptlock_s *get_shm_lock(int id) {

	void *shm;
	cktptlock_s *lock;
	int flag = 0;

	flag = 0666;


	shm = get_shm(id,flag);
	if(!shm)
		return NULL;

	lock = (cktptlock_s *)shm;
	
	return lock;
}



int init_shm_lock(int id) {

    if(g_chkptlock)
		return 0;

	g_chkptlock = create_shm_lock(id);
	g_chkptlock->siglist = -1;
	assert(g_chkptlock);
	gt_spinlock_init(&g_chkptlock->lock);
	return 0;
}

int set_acquire_chkpt_lock(int id) {

	if(!g_chkptlock) {
		return -1;
	}

	 gt_spin_lock(&g_chkptlock->lock);

	return 0;
}




int send_lock_avbl_sig(int signo) {

	if(g_chkptlock->siglist >= 0){
#ifdef NV_DEBUG
		fprintf(stdout,"success sent signal to pid %d\t"
			"rank %d \n", g_chkptlock->siglist,
			 g_chkptlock->rank);	
#endif
	 	kill(g_chkptlock->siglist,signo);
	}
	else{
		//fprintf(stdout,"send signal failed\n");
		return -1;
	}
}



/*SIGNAL handling functions */
int register_ckpt_lock_sig(int pid, int signo) {


    g_chkptlock = get_shm_lock(pid);

	if(!g_chkptlock)
		return -1;

	g_chkptlock->siglist = getpid();
	g_chkptlock->rank = pid;

#ifdef NV_DEBUG
	fprintf(stderr,"registering for signal %d \n", g_chkptlock->siglist);
#endif

	/*now register for signal*/
	if (signal(signo, sig_remote_chkpt) == SIG_ERR){
		 printf("signal(SIGINT) error\n");
		 exit(1);
	 }

	return 0;
}


#ifdef _RMT_PRECOPY
	int prev_chkpt_cnt = 0;
#endif


/*all these var should be convered to global_cur
 * procid*/
int term_proc_id = -1;
static void set_term_flag(int signo) {

	 size_t chkpt_sz;

	//fprintf(stdout,"APP FINISH INSTR RECVD\n");
	copy_proc_checkpoint(term_proc_id, &chkpt_sz, 0, SYNC_LCL_CKPT);
#ifdef _BW_PROFILING
	print_all_bw_timestamp();
#endif

}


int app_exec_finish_sig(int pid, int signo) {

	term_proc_id = pid;
	/*now register for signal*/
	if (signal(signo, set_term_flag) == SIG_ERR){
		 printf("signal(SIGINT) error\n");
		 exit(1);
	 }
	 return 0;
}


int rmt_chkpt_cnt =0;

int wait_for_chkpt_sig(int pid, int signo) {

   volatile int flg =0;
   volatile int chkpt_cnt = 0;	
   int rem;

 
	sigemptyset(&newmask);
	sigaddset(&newmask, signo);
	sigprocmask(SIG_BLOCK, &newmask, NULL);
#ifdef NV_DEBUG
	 fprintf(stdout,"waiting for signal rankid %d,"
			 "mypid %d\n", pid, getpid());
#endif
	sigwait(&newmask, &signo);
#ifdef NV_DEBUG
	 fprintf(stdout,"got signal\n");
#endif

	if(!g_chkptlock)
    g_chkptlock = get_shm_lock(pid);

    if(!g_chkptlock)
        return NO_SHM_AVBL;

	//sleep(1);
	//return START_CKPT;	
	chkpt_cnt = g_chkptlock->local_chkpt_cnt;

#ifdef _RMT_PRECOPY
	if(chkpt_cnt && prev_chkpt_cnt == chkpt_cnt) {
		//fprintf(stdout, "prev_chkpt_cnt %d,chkpt_cnt %d\n",
        //        prev_chkpt_cnt, chkpt_cnt);

		return INCORR_INTERVAL;
	}
	else {
		prev_chkpt_cnt = chkpt_cnt;
	}

#endif


#ifdef NV_DEBUG
    fprintf(stdout, "local_chkpt_cnt %d,REMOTE_FREQUENCY %d\n",
                         chkpt_cnt, REMOTE_FREQUENCY);
#endif

	rem = chkpt_cnt % REMOTE_FREQUENCY;

#ifdef _RMT_PRECOPY

   if((REMOTE_FREQUENCY - rem <= 2)){
		//fprintf(stdout, "local_chkpt_cnt %d,REMOTE_FREQUENCY %d\n",
        //        chkpt_cnt, REMOTE_FREQUENCY);
		if(REMOTE_FREQUENCY - rem == 2) {
			//long  cmt_freq = get_rmt_commit_freq();
			//sleep(30);
		}
		rmt_chkpt_cnt++;
		return START_CKPT;
	}else{

		return INCORR_INTERVAL;
	}
#else
	if( !rem) {
		//fprintf(stdout, "local_chkpt_cnt %d,REMOTE_FREQUENCY %d\n",
        //        chkpt_cnt, REMOTE_FREQUENCY);
		return START_CKPT;
	}
	else {
      //fprintf(stdout, "local_chkpt_cnt %d,REMOTE_FREQUENCY %d\n",
      //                  chkpt_cnt, REMOTE_FREQUENCY);
		return INCORR_INTERVAL;
	}
#endif

	/*while(!flg)	{
		flg = g_chkptlock->dirty;
#ifdef NV_DEBUG
	   fprintf(stdout,"waiting for dirty bit set \n");
#endif
	}
#ifdef NV_DEBUG
    fprintf(stdout,"dirty bit set \n");
#endif*/

}


int set_chkpt_type(int type){

	if(!g_chkptlock){
		return -1;
	}
	g_chkptlock->chkpt_type = type;
	return 0;
}

int get_chkpt_type(){

	volatile int chkpt_type =0;

	if(!g_chkptlock){
		return -1;
	}
	chkpt_type = g_chkptlock->chkpt_type;	

	return chkpt_type;
}


int incr_lcl_chkpt_cnt() {

	if(!g_chkptlock){
		return -1;
	}

	g_chkptlock->local_chkpt_cnt++;

	return 0;
}

int acquire_chkpt_lock(int id) {

	if(!g_chkptlock){
		return -1;
	}

	gt_spin_lock(&g_chkptlock->lock);

	return 0;
}


int disable_chkpt_lock(int id) {

    if(g_chkptlock);
    gt_spin_unlock(&g_chkptlock->lock);
	return 0;
}

int set_ckptdirtflg(int id) {

	assert(g_chkptlock);
	g_chkptlock->dirty = 1;
	return 0;
}

int disable_ckptdirtflag(int id) {

    assert(g_chkptlock);
    g_chkptlock->dirty = 0;
	return 0;
}

int get_dirtyflag( int id) {

	volatile int flg =0;

	if(!g_chkptlock) {
		fprintf(stdout,"shared mem not created %d\n",
				id);
		return 0;
	}

#ifdef NV_DEBUG
	fprintf(stdout,"get_dirtyflag->dirty %d\n",
			g_chkptlock->dirty);
#endif

	flg = g_chkptlock->dirty;

	return flg;
}

static void sig_remote_chkpt(int signo)
{

	//sig_queue.push(signo);
#ifdef NV_DEBUG
	fprintf(stdout, "remote signal recived %d\n", signo);
#endif
   // sigflag = 1;
	/*now register for signal*/
	if (signal(signo, sig_remote_chkpt) == SIG_ERR){
		 printf("signal(SIGINT) error\n");
		 exit(1);
	}	 
    return;
}


#ifdef _USE_FAULT_PATTERNS

int add_chunk_fault_lst(int chunkid){

	chunk_fault_list.push_back(chunkid);
	return 0;
}


int find_chunk_fault_list(int chunkid) {


	std::list<int>::iterator it;

	it = std::find(chunk_fault_list.begin(),chunk_fault_list.end(), chunkid);
	if(it == chunk_fault_list.end())
		return -1;
	else 
		return 0;

}

int find_nxtchunk_faultlst(int chunkid) {


	std::list<int>::iterator it;

    	it = std::find(chunk_fault_list.begin(),chunk_fault_list.end(), chunkid);
	it++;

	if(it == chunk_fault_list.end())
		return 0;
	else 
		return *it;

}



int get_next_chunk(int chunkid) {


	//return find_nxtchunk_faultlst(chunkid);
	int id =0;
	if(!chunk_fault_list.size())
		return -1;
	id = chunk_fault_list.front();
	chunk_fault_list.pop_front();
	chunk_fault_list.push_back(id);
	return id;
}

int check_chunk_fault_lst_empty() {

	if(chunk_fault_list.empty())
		return 1;
	else
		return 0;

}

int print_chunk_fault_lst() {

	std::list<int>::iterator itr;
	

	fprintf(stdout,"\n\n\n\n");	


	for(itr=chunk_fault_list.begin(); 
			itr != chunk_fault_list.end(); 
			itr++) {

		fprintf(stdout,"%d->",*itr);	
	}
	fprintf(stdout,"\n\n\n\n");	

	return 0;
}

#endif

int set_protection(void *addr, size_t len, int flag){


   //very dangerous code
   unsigned long off = 0;
   size_t protect_len =0;
   off = (4096- ((unsigned long)addr % 4096));
   addr = addr + off;
   protect_len = len - off;
   off = protect_len % 4096;      
   protect_len = protect_len -off;

    if (mprotect(addr,protect_len, flag)==-1) {
		if(prev_proc_id ) {
			fprintf(stdout,"%lu len %u\n", (unsigned long)addr, len);
	    	perror("mprotect");
			exit(-1);	
		}
    }
#ifdef NV_DEBUG
	if(prev_proc_id == 1)
		fprintf(stdout,"setting protection \n"); 	
#endif
	return 0;
}

size_t remove_chunk_prot(void *addr, int *chunkid) {

	chunkobj_s *chunk = NULL;
	chunk =(chunkobj_s *)get_chunk_from_map(addr); 
	
	/*if(!chunk) {
        unsigned long off = 0;
       	off =  (((unsigned long)addr % 4096));
		fprintf(stdout,"off %d \n", off);
		if(off){
		     set_protection(addr-off,4096,PROT_READ|PROT_WRITE);
		     fprintf(stdout, "finished protection\n");	
			return 4096;
		}else
			assert(off);
	}*/ 	
	if(!chunk) {
		if(prev_proc_id) {
			fprintf(stdout, "chunk error %lu proc %d\n",
				(ULONG)addr, prev_proc_id);
#ifdef _NVSTATS
			print_all_chunks();
#endif
		}
	}

	assert(chunk);
	chunk->chunk_commit = 0;
	chunk->dirty = 1; 
	//chunk->rmt_nvdirtchunk =1;
	*chunkid = chunk->chunkid;

	//if(blablabla)
#ifdef _USE_FAULT_PATTERNS
//	add_chunk_fault_lst(chunk->chunkid);
#endif
	
	//if(prev_proc_id == 1)
	//fprintf(stdout,"chunk:%d--->",chunk->chunkid);
	//fprintf(stdout,"fault chunk:%d \n",chunk->chunkid);

	set_protection(chunk->log_ptr,chunk->logptr_sz, 
					PROT_READ|PROT_WRITE);

	return chunk->length;
}

int set_chunkprot_using_map() {

	chunkobj_s *chunk;

#ifdef NV_DEBUG
	fprintf(stdout, "enabling chunk protection \n");
#endif

	if(!chunkmap.size())
		return -1;

    for( chunk_itr= chunkmap.begin(); 
			chunk_itr!=chunkmap.end(); 
			++chunk_itr){

        chunk = (chunkobj_s *)(*chunk_itr).second;
		assert(chunk);

	    assert(chunk->nv_ptr);
    	assert(chunk->log_ptr);
	    /*if(chunk->length < 1000000){
		 chunk->dirty = 1;	
         continue;
	   }*/
	   	set_protection(chunk->log_ptr,
                    chunk->logptr_sz,
                    PROT_READ);
		 chunk->dirty = 0;
    }
    return 0;
}



int enabl_chunkprot_using_map(int chunkid) {

	 chunkobj_s *chunk;

    chunk = (chunkobj_s *)id_chunk_map[chunkid];
    assert(chunk);

    /*if(chunk->length < 1024 * 1024) {
      chunk->dirty = 1;
      return 0;
    }*/
    assert(chunk->nv_ptr);
    assert(chunk->log_ptr);

    set_protection(chunk->log_ptr,chunk->logptr_sz,PROT_READ);

	return 0;

}


#if 0
int enabl_exclusv_chunkprot(int chunkid) {

	chunkobj_s *chunk;

	if(!chunkmap.size())
		return -1;

    for( chunk_itr= chunkmap.begin(); 
			chunk_itr!=chunkmap.end(); 
			++chunk_itr){

        chunk = (chunkobj_s *)(*chunk_itr).second;
		assert(chunk);
		if(chunk->chunkid ==chunkid)
			continue;

		if(chunk->dirty)
			 continue;

		
        /*if(chunk->length < 1000000){
	         chunk->dirty = 1;  
    	     continue;
        } */   

		assert(chunk->nv_ptr);
		assert(chunk->log_ptr);
		//if(prev_proc_id == 1)
		//fprintf(stdout,"protected address %lu %d\n",(ULONG)chunk->log_ptr, chunk->chunkid);
		set_protection(chunk->log_ptr,
                    chunk->logptr_sz,
                    PROT_READ);
		//chunk->chunk_commit = 0;
    }
    return 0;
}
#endif
#if 0
int copy_dirty_chunks() {

	chunkobj_s *chunk;

    for( chunk_itr= chunkmap.begin(); 
			chunk_itr!=chunkmap.end(); 
			++chunk_itr){

        chunk = (chunkobj_s *)(*chunk_itr).second;
		assert(chunk);

		if(!chunk->dirty)
			continue;

		if(chunk->length < MIN_ASYNC_DATA_LEN)
			continue;

		assert(chunk->nv_ptr);
		assert(chunk->log_ptr);
		chunk->dirty = 0;
		memcpy_delay(chunk->nv_ptr,chunk->log_ptr, chunk->length);

			set_protection(chunk->log_ptr,chunk->logptr_sz,PROT_READ);
    }

    return 0;
}
#endif

int copy_dirty_chunk(int chunkid, int memcpy_flg) {

	chunkobj_s *chunk;

	chunk = (chunkobj_s *)id_chunk_map[chunkid];
	assert(chunk);

	if(!chunk->dirty)
		return 0;	
	if(chunk->length < MIN_ASYNC_DATA_LEN) {
	  chunk->dirty = 1;
	  return 0;
    }	

	if(memcpy_flg) {
		assert(chunk->nv_ptr);
		assert(chunk->log_ptr);
		chunk->dirty = 0;
#ifdef _ASYNC_RMT_CHKPT
	//	if(chunk->length > MIN_ASYNC_RMT_DATA_LEN) {
			chunk->rmt_nvdirtchunk = 1;
	//	}
#endif

#ifdef _NVSTATS
		add_to_chunk_memcpy(chunk);
#endif
		memcpy_delay_temp(chunk->nv_ptr,chunk->log_ptr, chunk->length);
		//memcpy(chunk->nv_ptr,chunk->log_ptr, chunk->length);
	}else {

	  chunk->dirty = 0;	
	}

//#ifdef NV_DEBUG
	if(prev_proc_id == 1)
	fprintf(stdout,"async LC copied Chnk %d %d\n", chunk->chunkid, prev_proc_id);
//#endif

	/*unsigned long off = 0;
	off = (((unsigned long)chunk->logptr_sz % 4096));
	size_t len = chunk->logptr_sz - off - 4096;*/
	size_t len = chunk->logptr_sz;
	set_protection(chunk->log_ptr,len,PROT_READ);
#ifdef NV_DEBUG
	if(prev_proc_id == 1)
		fprintf(stdout,"setting protection \n"); 	
#endif
    return 0;
}



#ifdef _ARMCI_CHECKPOINT
int add_chkpt_ver(void *buffer, size_t size, int rank, int version ) {

	assert(buffer);
	sprintf((char *)buffer,"%d:%d",rank,version);

	return 0;
}

int debug_chkpt_ver(void **buffer, size_t size, int rank, int version) {

	void *ptr;
	int peer;

    if(rank == 0) {
		peer =  (rank);
		assert(buffer[peer]);
		ptr = buffer[peer]; 
		fprintf(stdout, "my rank %d, ptr %s \n", rank, ptr);
	}
	return 0;
}

#endif

void* copy_proc_checkpoint(int procid, size_t *chkpt_sz,
						   int check_dirtpages,
						   int ckpttype) {

    int numpages = 0;
    size_t bytes = 0, total_size=0, sent_bytes =0;
	size_t header_sz =0;
    void *sendbuff = 0, *tmp = NULL;
    int type =0, vmaid =0, mapid = 0;


    if(!proc_vmas.size()){
        perror("no vmas added yet \n");
        return NULL;
    }

#ifdef _NVSTATS

	 //if(ckpttype == SYNC_LCL_CKPT)
		rstat.num_chkpts++;

    /*r_end measures interval*/
	gettimeofday(&r_end, NULL);
    /*r_ckpt_start measures per step time*/
	r_ckpt_start = r_end;

	if(rstat.num_chkpts > 1)
	rstat.commit_freq =simulation_time(r_start,r_end);   
#endif

	header_sz = sizeof(chkpthead_s);

   /*for( vma_itr= metadata_vmas.begin(); vma_itr!= metadata_vmas.end(); ++vma_itr){

		vmaid = (*vma_itr).first;
		bytes = (*vma_itr).second;

		total_size = total_size + bytes + header_sz;
		sendbuff = realloc(sendbuff,total_size);
		assert(sendbuff);

		tmp = helper_update_sendbuff(sendbuff,
							tmp,bytes, &sent_bytes,
							procid, type, vmaid);

       copy_dirty_pages(procid,vmaid,tmp, bytes);
	   sent_bytes =  sent_bytes + bytes + header_sz;	

	}*/

    for( chunk_itr= chunkmap.begin(); chunk_itr!=chunkmap.end(); ++chunk_itr){
        chunkobj_s *chunk = (chunkobj_s *)(*chunk_itr).second;
        bytes = chunk->length;  
		mapid = chunk->vma_id;
		vmaid = chunk->chunkid;
        assert(bytes);

	    chunk->chunk_commit = 1;

		/*copy only if this chunk in nvram is not dirty*/
		if(!chunk->rmt_nvdirtchunk) {
			continue;
		}


#ifdef _ARMCI_CHECKPOINT
		if(chunk->rmt_armci_ptr) {
			chunk->version++;	

			assert(chunk->dummy_ptr);
			//add_chkpt_ver(ptr,chunk->length, chunk->my_rank, chunk->version);
			if(procid == 1) {
			//if(ckpttype == ASYNC_LCL_CKPT)
				fprintf(stdout, "chunk->rmt_nvdirtchunk %d "
                    "chunkid %d num_rmt_cpy %d size %u\n",
               	     chunk->rmt_nvdirtchunk,
                   	 chunk->chunkid, chunk->num_rmt_cpy,
					 chunk->length);
			}	
		
			armci_remote_memcpy(chunk->my_rank, chunk->my_rmt_chkpt_peer, 	
						chunk->rmt_armci_ptr, chunk->dummy_ptr, bytes);
			 chunk->rmt_nvdirtchunk = 0;
			//invoke_barrier();
			//debug_chkpt_ver(chunk->rmt_armci_ptr, chunk->length,  chunk->my_rank, chunk->version); 
		 }
#else
		if(procid == 1)
       fprintf(stdout, "procid %d chunk->chunkid %d, length %u\n",
					procid, chunk->chunkid,  chunk->length);    

        total_size = total_size + bytes + header_sz;
        sendbuff = realloc(sendbuff,total_size);
        assert(sendbuff);

		tmp = helper_update_sendbuff(sendbuff,
							tmp,bytes, &sent_bytes,
							procid, type, vmaid);

        memcpy(tmp,chunk->rmt_log_ptr, bytes);
		//if(procid == 1)
	      // fprintf(stdout, "procid %dunk->chunkid %d \n",
			//		procid, chunk->chunkid);    
       //copy_dirty_pages(procid,mapid,tmp, bytes);
		//assert(chunk->nv_ptr);
		//memcpy(tmp, chunk->nv_ptr, bytes);
	   sent_bytes =  sent_bytes + bytes + header_sz;	
#endif
		chunk->num_rmt_cpy++;
		chunk->rmt_nvdirtchunk = 0;

#ifdef _NVSTATS
		rstat.chkpt_chunks++;
		rstat.chkpt_sz += bytes;
#endif

    }
	//vma_itr= proc_vmas.begin();
    //int mapid = (*vma_itr).first;
   /*for( vma_itr= proc_vmas.begin(); vma_itr!=proc_vmas.end(); ++vma_itr){

        vmaid = (*vma_itr).first;
        if(check_dirtpages) {

            numpages = get_vma_dirty_pgcnt(procid, vmaid);
            bytes =numpages * PAGE_SIZE;
            fprintf(stdout, "procid: %d vmaid: %d, "
                    "dirtypages %d header_sz %u\n",
                    procid, vmaid,numpages, header_sz);
        }else{

            bytes = (*vma_itr).second;
        }
        assert(bytes);
   	    if(!bytes)
       	    continue;

        total_size = total_size + bytes + header_sz;
        sendbuff = realloc(sendbuff,total_size);
        assert(sendbuff);

		tmp = helper_update_sendbuff(sendbuff,
							tmp,bytes, &sent_bytes,
							procid, type, vmaid);

       copy_dirty_pages(procid,vmaid,tmp, bytes);
	   sent_bytes =  sent_bytes + bytes + header_sz;	
    }*/
    *chkpt_sz = total_size;
	//parse_data(sendbuff, total_size);
	
#ifdef _NVSTATS
    gettimeofday(&r_start, NULL);
	r_ckpt_end = r_start;
	rstat.chkpt_time =simulation_time(r_ckpt_start,r_ckpt_end);
	//print_rmt_chkpt_stats()
	if(procid == 1)
	fprintf(stdout,"chkpt_sz/chkpt %u\n",rstat.chkpt_sz);
	rstat.chkpt_sz = 0;
#endif

#ifdef _ARMCI_CHECKPOINT
	invoke_barrier();
#endif

	return sendbuff;
}

/*------------------------------------------------------------*/

#ifdef _ARMCI_CHECKPOINT

int my_rmt_peer;

int register_chkpt_peer(int peer) {

	my_rmt_peer = peer;

	return 0;
}

/*Currently code supports only one peer*/
int setup_memory(int nranks, int myrank) {

	chunkobj_s *chunk;
	size_t u_bytes;
	int members[2];	

	if(myrank > my_rmt_peer){
		members[0] = my_rmt_peer;
		members[1] = myrank;
	}else{
		members[1] = my_rmt_peer;
		members[0] = myrank;
	}	

	/*first create a group*/
	create_group (members, 2, myrank,  nranks);

    for( id_chunk_itr= id_chunk_map.begin();
            id_chunk_itr !=id_chunk_map.end();
            ++id_chunk_itr){

        chunk = (chunkobj_s *)(*id_chunk_itr).second;
        assert(chunk);
        u_bytes = chunk->length + ERROR_BYTES;

		/*the last byte of every chunk would
		contain version info*/
		//u_bytes += VERSION_INFO_SZ;
		//fprintf(stdout, "registering %u %u\n",u_bytes, ERROR_BYTES);

		//if(myrank == 0)
		//chunk->rmt_armci_ptr = create_memory(nranks, myrank, u_bytes);
		chunk->rmt_armci_ptr = NULL;
		chunk->rmt_armci_ptr= group_create_memory(nranks, myrank, u_bytes);
		//fprintf(stdout, "finished creating chunk %d %u %d\n", chunk->chunkid, u_bytes, cntr);
		assert(chunk->rmt_armci_ptr);
		chunk->dummy_ptr = malloc(u_bytes);
		//if(myrank == 0)
		//	fprintf(stdout, "finished creating chunk %d\n", chunk->chunkid);
		//set the remote checkpoint peer;
	    chunk->my_rmt_chkpt_peer = my_rmt_peer;
        chunk->my_rank = myrank;

    }
    return 0;
}
    


#endif


#ifdef _NVSTATS
int print_all_chunks() {

	chunkobj_s *chunk;
	size_t bytes = 0;
    unsigned long start, end;
	
    //for( chunk_itr= chunkmap.begin(); chunk_itr!=chunkmap.end(); ++chunk_itr){
    for( chunk_itr=  chunkmap.begin(); chunk_itr!= chunkmap.end(); ++chunk_itr){
		chunk = (chunkobj_s *)(*chunk_itr).second;
        bytes = chunk->logptr_sz;
		start = (ULONG)(*chunk_itr).first;
		end = start + bytes;
		fprintf(stdout,"chunk:id %d\t"
				"memcpys: %d\t"
				"chunk size: %d\n", 	
				chunk->chunkid,
				chunk->num_memcpy,
				chunk->length);
				/*"start %lu \t end %lu \t"
				"chunk->logptr_sz %u\n",
				 chunk->chunkid,start, end,
				 chunk->logptr_sz);*/
	}
	return 0;
}


int print_adtnl_copy_overhead(int num_chkpts) {

	 chunkobj_s *chunk;
    unsigned long  bytes = 0;
	unsigned long extra_copy_bytes =0;
	unsigned long  actual_bytes = 0;
	unsigned long  additional_bytes = 0;
	unsigned long  reduced_bytes =0;

    for( chunk_itr= chunkmap.begin(); chunk_itr!=chunkmap.end(); ++chunk_itr){
        chunk = (chunkobj_s *)(*chunk_itr).second;


		if(chunk->length) {

			bytes += (chunk->length *  chunk->num_memcpy);
			actual_bytes += (chunk->length * num_chkpts);
		}
    }

	/*if( bytes > actual_bytes)
		additional_bytes = (long)(bytes- actual_bytes);
	else
		reduced_bytes = (long)(actual_bytes- bytes);*/
	fprintf(stdout, "ORIGINIAL COPY/proc (bytes) %lu\n",actual_bytes);
	fprintf(stdout, "OBSERVED COPY/proc (bytes) %lu\n",bytes);

	if(additional_bytes)
		fprintf(stdout, "ADDITIONAL COPY/proc (bytes) %lu\n",additional_bytes);
	else
		fprintf(stdout, "REDUCED COPY/proc (bytes) %lu\n",reduced_bytes);

	/*if(additional_bytes)
    	fprintf(stdout, "ADDITIONAL COPY PERCENT %f\n",(float)((((float)bytes/(float)actual_bytes) *100) - 100));
	else
		fprintf(stdout, "REDUCED COPY PERCENT %f \n", (float)((((float)actual_bytes/(float)bytes) *100) - 100));*/
    return 0;

}

long get_rmt_commit_freq(){

	return rstat.commit_freq;
}

void clear_rmt_stats(){
	//rstat.num_chkpts = 0;
	rstat.chkpt_chunks =0;
	rstat.chkpt_sz =0;
	rstat.commit_freq =0;
	rstat.chkpt_time = 0;
}

int print_rmt_chkpt_stats( int pid, int myrank) {

	if(myrank == 0){ 
		fprintf(stdout,"No. of checkpoint %d\n",rstat.num_chkpts);
		fprintf(stdout,"Num. Chkpt chunks %d\n",rstat.chkpt_chunks);
		fprintf(stdout,"chkpt_sz/chkpt %u\n",rstat.chkpt_sz);
		fprintf(stdout,"Chkpt interval %ld\n",rstat.commit_freq);
		fprintf(stdout,"Time/chkpt(sec) %ld\n",rstat.chkpt_time/1000000);
	}
	/*clear all stats*/
	clear_rmt_stats();	

	return 0;
}

#endif
#endif


//#ifdef _USE_FAULT_PATTERNS

int add_chunk_fault_lst(int chunkid){

	chunk_fault_list.push_back(chunkid);
	return 0;
}


int find_chunk_fault_list(int chunkid) {

	std::list<int>::iterator it;

	it = std::find(chunk_fault_list.begin(),chunk_fault_list.end(), chunkid);
	if(it == chunk_fault_list.end())
		return -1;
	else 
		return 0;

}

int find_nxtchunk_faultlst(int chunkid) {


	std::list<int>::iterator it;

    	it = std::find(chunk_fault_list.begin(),chunk_fault_list.end(), chunkid);
	it++;

	if(it == chunk_fault_list.end())
		return 0;
	else 
		return *it;

}

int get_next_chunk(int chunkid) {
	//return find_nxtchunk_faultlst(chunkid);
	int id =0;
	if(!chunk_fault_list.size())
		return -1;
	id = chunk_fault_list.front();
	chunk_fault_list.pop_front();
	chunk_fault_list.push_back(id);
	return id;
}

int check_chunk_fault_lst_empty() {

	if(chunk_fault_list.empty())
		return 1;
	else
		return 0;
}

int print_chunk_fault_lst() {

	std::list<int>::iterator itr;
	fprintf(stdout,"\n\n\n\n");	

	for(itr=chunk_fault_list.begin(); 
			itr != chunk_fault_list.end(); 
			itr++) {

		fprintf(stdout,"%d->",*itr);	
	}
	fprintf(stdout,"\n\n\n\n");	

	return 0;
}

//#endif
int set_chunk_protection_ulong (unsigned long addr, size_t len, int flag){

   unsigned long off = 0;
   size_t protect_len =0;
   off = (4096- (addr % 4096));
   addr = addr + off;
   protect_len = len - off;
   off = protect_len % 4096;      
   protect_len = protect_len -off;

	fprintf(stderr,"in setting protection %d, addr %lu, len %u\n", flag, addr, protect_len); 	
    if (mprotect((void *)addr,protect_len, flag)==-1) {
		if(prev_proc_id ) {
			fprintf(stdout,"%lu len %u\n", (unsigned long)addr, len);
	    	perror("mprotect");
			exit(-1);	
		}
    }
	return 0;
}

int set_chunk_protection(void *addr, size_t len, int flag){

   unsigned long off = 0;
   size_t protect_len =0;
   off = (4096- ((unsigned long)addr % 4096));
   addr = addr + off;

   protect_len = len - (len % 4096);
   //fprintf(stderr,"in setting protection %d, addr %lu, len %u\n", flag, addr, protect_len); 	
    if (mprotect(addr,protect_len, flag)==-1) {
	    perror("mprotect");
		exit(-1);	
    }
	return 0;
}

size_t remove_chunk_prot(void *addr, int *chunkid) {

	chunkobj_s *chunk = NULL;
	chunk =(chunkobj_s *)get_chunk_from_map(addr); 

	if(!chunk) {
		if(prev_proc_id) {
			fprintf(stdout, "chunk error %lu proc %d\n",
				(ULONG)addr, prev_proc_id);
#ifdef _NVSTATS
			print_all_chunks();
#endif
		}
	}
	assert(chunk);
	//chunk->chunk_commit = 0;
	//chunk->dirty = 1; 
	*chunkid = chunk->chunkid;

#ifdef _USE_FAULT_PATTERNS
//	add_chunk_fault_lst(chunk->chunkid);
#endif
	set_chunk_protection(chunk->nv_ptr,chunk->length, 
					PROT_READ|PROT_WRITE);
	return chunk->length;
}

size_t nv_disablprot(void *addr, int *curr_chunkid) {
    int chunkid = 0;
    size_t length = 0;

	fprintf(stdout,"disabling protection \n");
    length =remove_chunk_prot(addr, &chunkid);
    *curr_chunkid = chunkid;

#ifdef _USE_FAULT_PATTERNS
    int nxt_chunk=0;
    //if(find_chunk_fault_list(chunkid)){   
    if(!chunk_fault_lst_freeze)
        add_chunk_fault_lst(chunkid);

    else {
        /*get the next chunk to be protected
        set the protection of the chunk*/
        nxt_chunk = get_next_chunk(chunkid);
        if(!nxt_chunk)
            goto end;

        if(!nxt_chunk || enable_chunkprot(nxt_chunk))
            if(prev_proc_id == 1)
                fprintf(stderr,"chun protection failed\n");
    }
#endif
    end:
    return length;
}

/*int set_chunkprot_using_map() {

	chunkobj_s *chunk;

#ifdef NV_DEBUG
	fprintf(stdout, "enabling chunk protection \n");
#endif

	if(!chunkmap.size())
		return -1;

    for( chunk_itr= chunkmap.begin(); 
			chunk_itr!=chunkmap.end(); 
			++chunk_itr){

        chunk = (chunkobj_s *)(*chunk_itr).second;
		assert(chunk);

	    assert(chunk->nv_ptr);
	   	set_protection(chunk->nv_ptr,
                    chunk->length,
                    PROT_READ);
		 chunk->dirty = 0;
    }
    return 0;
}*/



/*int enabl_chunkprot_using_map(int chunkid) {

	 chunkobj_s *chunk;

    chunk = (chunkobj_s *)id_chunk_map[chunkid];
    assert(chunk);

    assert(chunk->nv_ptr);
    set_protection(chunk->nv_ptr,chunk->length,PROT_READ);

	return 0;

}*/


