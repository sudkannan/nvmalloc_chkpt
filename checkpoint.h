#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "nv_map.h"
#include <sys/ipc.h>
#include <sys/shm.h>
#include "hash_maps.h"


#define MAP_SIZE 1024 * 10
#define PAGE_SIZE 4096
#define INTERGER_BUFF 100000
#define ERROR_BYTES 1024*1024*2
#define SHMID 9999
#define MIN_ASYNC_RMT_DATA_LEN 1*1024*1024

void* copy_proc_checkpoint(int procid, size_t *chkpt_sz,
						   int check_dirtpages,
						   int ckpttype);

int parse_data(void *buffer, size_t size);

/*checkpoint lock methods
these are per process locks
and do not support threads now*/
int set_acquire_chkpt_lock(int id);
int acquire_chkpt_lock(int id);
int disable_chkpt_lock(int id);

int set_ckptdirtflg(int id);
int disable_ckptdirtflag(int id);
int get_dirtyflag(int id);
int register_ckpt_lock_sig(int pid, int signo); 
int wait_for_chkpt_sig(int signo, int pid);
int  send_lock_avbl_sig(int signo);
int init_shm_lock(int id);
int set_chunk_write_prot() ;
size_t remove_chunk_prot(void *addr, int *chunkid);
int set_chunkprot_using_map();


int get_chnk_cnt_frm_map();
void* get_chunk_with_id(UINT chunkid);

int enabl_chunkprot_using_map(int chunkid);
int app_exec_finish_sig(int pid, int signo);

//Reducing checkpoint contention
int add_chunk_fault_lst(int chunkid);
int get_next_chunk(int chunkid);
int print_chunk_fault_lst();
int find_chunk_fault_list(int chunkid);
int check_chunk_fault_lst_empty();

int copy_dirty_chunks();
int copy_dirty_chunk(int chunkid, int memcpy_flg);
int enabl_exclusv_chunkprot(int chunkid);

int setup_memory(int nranks, int myrank);
int register_chkpt_peer(int);

int print_all_chunks();
int incr_lcl_chkpt_cnt();
int set_chkpt_type(int type);
int get_chkpt_type();
long get_rmt_commit_freq();
int set_chunk_protection(void *addr, size_t len, int flag);
int set_chunk_protection_ulong (unsigned long addr, size_t len, int flag);

#ifdef _NVSTATS
int print_adtnl_copy_overhead(int num_chkpts);
int print_rmt_chkpt_stats(int pid,int rank);
#endif

/*void* get_chunk_from_map(void *addr);
void* get_chunk_with_id(UINT chunkid);
int record_chunks(void* addr, chunkobj_s *chunk);
int get_chnk_cnt_frm_map();
void *get_chunk_from_map_o1(void *addr);
int record_vmas(int vmaid, size_t size);
int record_metadata_vma(int vmaid, size_t size);
size_t get_vma_size(int vmaid);*/
#endif
