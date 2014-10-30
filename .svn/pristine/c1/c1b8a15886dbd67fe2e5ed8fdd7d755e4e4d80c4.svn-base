#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <inttypes.h>
#include "pin_mapper.h"
#include <iostream>

using namespace std;

static void *pin_map = NULL;
static FILE *fp = NULL;
void *shm_addr = NULL;

#define NVM_ADDR_LST "/mnt/pmfs/nvaddrlist"

struct address{
	uint32_t start;
	uint32_t end;
};
typedef struct address s_addr;



void* CreateMapFile(char *filepath, unsigned long bytes) {

    int result;
    int fd = -1;
    void *addr = NULL;

    fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600);
    if (fd == -1) {
        perror("Error opening file for writing");
        exit(-1);
    }

    result = lseek(fd,bytes, SEEK_SET);
    if (result == -1) {
        close(fd);
        perror("Error calling lseek() to 'stretch' the file");
        exit(-1);
    }

    result = write(fd, "", 1);
    if (result != 1) {
        close(fd);
        perror("Error writing last byte of the file");
        exit(-1);
    }
    addr = mmap(0,  bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "mmap failed in webshoot\n");
        close(fd);
        exit(-1);
   }

    pin_map =  addr;
    //fp = fmemopen(pin_map, 1024*1024, "w");

    return addr;
}

char * CreateSharedMem() {

    int shmid;
    key_t key;

    key = PIN_SHM_KEY;
	if(shm_addr) 
		return (char *)shm_addr;

   //if(pin_map)
    //	return pin_map;
    if ((shmid = shmget(key, PIN_SHM_SZ, IPC_CREAT | 0666)) < 0) {
        perror("shmget : in webshoot");
        exit(1);
    }
    if ((shm_addr = shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
        exit(1);
    }
    pin_map = (void *)shm_addr;
    if(pin_map) fprintf(stdout,"created sharedmemory\n");
    pin_map = shm_addr;
    //fp = fmemopen(shm_addr, 1024*1024, "w");
    //assert(fp);
    return (char *)shm_addr;
}

int idx=0;
s_addr *addr=NULL;

char* Writeline(unsigned long strt, unsigned long end){

	if(!pin_map) {
		CreateMapFile((char *)NVM_ADDR_LST, 1024*1024);
		fprintf(stderr,"Created PIN MAP file \n");
		assert(pin_map);
		addr = (s_addr *)pin_map;
	}
	//s_addr *current = (s_addr *)pin_map;
    //s_addr *addr = (s_addr *)current;
	addr[idx].start = (uint32_t)strt;
	addr[idx].end = (uint32_t)end;
    std::cout<<"LoadNVMAddress  " << addr[idx].start<<" "<< addr[idx].end << endl;
    idx++;	
	//pin_map += sizeof(s_addr);
	/*sprintf(current,"%lu %lu", strt, end);
	current = current + 31;
	current[0] = '\n';
    //sprintf(pin_map,"%c",(char)'\n');
    pin_map = current;*/
	/*if(fp)
	fprintf (fp, "%lu %lu\n", strt, end);*/
    return (char *)pin_map;
}

/*char* Writeline(unsigned long strt, unsigned long end){

	if(!pin_map) {
		CreateSharedMem();
		assert(pin_map);
	}
	char *current = (char *)pin_map;

	sprintf(current,"%lu %lu", strt, end);
	current = current + 31;
	current[0] = '\n';
    //sprintf(pin_map,"%c",(char)'\n');
	current++;
    pin_map = current;
	//if(fp)
	//fprintf (fp, "%lu %lu\n", strt, end);
    return (char *)pin_map;
}*/
