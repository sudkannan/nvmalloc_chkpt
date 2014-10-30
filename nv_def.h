#ifndef NV_DEF_H_
#define NV_DEF_H_

#include <stdint.h>

#define __NR_nv_mmap_pgoff     314
//#define __NR_nv_mmap_pgoff     301

#define __NR_nv_commit		   303
#define __NR_copydirtpages 304

#define SHMEM_ID 2078
#define PAGE_SIZE 4096
#define CACHE_LINE_SIZE 64

typedef intptr_t nvword_t;
//This denotes the size of persistem memory mapping
// for each process. Note, the metadata mapping is a seperate
//memory mapped file for each process currently
#define PROC_METADAT_SZ 100*1024*1024
#define MMAP_METADATA_SZ 50*1024*1024
#define MAX_DATA_SIZE 1024 * 1024 *1524
#define NVRAM_DATASZ 1024 * 1024 * 500
#define BASE_METADATA_NVID 2777
#define BASEID 477799
#define TRANS_LOGSZ 500*1024*1024*1
#define TRANS_DATA_LOGSZ 500*1024*1024*1
#define TRANS_LOGVMAID 1000
#define TRANS_DATA_LOGVMAID 1001
#define PROT_NV_RW  PROT_READ|PROT_WRITE
#define PROT_NV_RDONLY  PROT_READ
#define PROT_ANON_PRIV MAP_PRIVATE | MAP_ANONYMOUS


//base name of memory mapped files
#define FILEPATH "/mnt/pvm/chkpt"
#define BASEPATH "/mnt/pvm/"
#define PROCMETADATA_PATH "/mnt/pvm/chkproc"
#define PROCLOG_PATH "/mnt/pvm/logproc"
#define PROCLOG_DATA_PATH "/mnt/pvm/logdataproc"
#define PROCMAPMETADATA_PATH "/mnt/pvm/chkprocmap"

/*maps all data when using disk*/
#define PROCMAPDATAPATH "/mnt/pvm/chkprocmapdata"

#define DATA_PATH "/mnt/pvm/procdata"

#define  SUCCESS 0
#define  FAILURE -1
//Page size
#define SHMSZ  4096 
//NVRAM changes
#define NUMINTS  (10)
#define FILESIZE (NUMINTS * sizeof(int))
//FIXME: UNUSED FLAG REMOVE
#define MMAP_FLAGS MAP_PRIVATE
//Enable debug mode
//#define NV_DEBUG

//Enable locking
#define ENABLE_LOCK 1

//Maximum number of process this library
//supports. If you want more proecess
//increment the count
#define MAX_PROCESS 64

//Random value generator range
//for temp nvmalloc allocation
#define RANDOM_VAL 1433
//#define NVRAM_OPTIMIZE

#define SHM_BASE 10000
#define CHKSUM_LEN 40 
#define REMOTE_FREQUENCY 2
#define THRES_ASYNC 18000000
#define SYNC_LCL_CKPT 2
#define ASYNC_LCL_CKPT 4
#define CACHELINESZ 64
#define MAX_FPATH_SZ 256

//PAGE Size
#define PAGESIZE 4096

#endif
