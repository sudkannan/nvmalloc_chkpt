#ifndef _NVMALLOC_H
#define _NVMALLOC_H

#ifdef __cplusplus
extern "C" {
#endif


void nvmalloc_init(unsigned nrPages, unsigned long freeWait);
void nvmalloc_exit(void);
void* nvmalloc(unsigned size);
int nvfree(void* addr);
void nvmalloc_print(void);
#ifdef __cplusplus
}
#endif

#endif
