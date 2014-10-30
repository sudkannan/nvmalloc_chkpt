#include <stdio.h>
#include <stdlib.h>
#include "nv_def.h"
#include "nv_stats.h"

#ifdef _LIBPMEMINTEL
//#include "pmem.h"	
#include "pmem_inline.h"
#endif


#define ASMFLUSH(dest) __asm__ __volatile__ ("clflush %0" : : "m"(*(volatile char *)dest))

static inline void clflush(volatile char* __p)
{
#ifdef _USE_CACHEFLUSH
    asm volatile("clflush %0" : "+m" (*__p));
#endif
    return;	
}


static inline void mfence()
{
    asm volatile("mfence":::"memory");
    return;
}

void flush_cache(void *ptr, size_t size){

  unsigned int  i=0;

#ifdef _LIBPMEMINTEL
	pmem_persist(ptr,size, 0);
#else
   //fprintf(stdout,"cache flush \n");
   //syscall(__NR_nv_commit, (unsigned long)ptr,
   //                    size, NULL);
  mfence();
  for (i =0; i < size; i=i+CACHELINESZ) {	

	//ASMFLUSH(ptr);
	clflush((volatile char*)ptr);
	ptr += CACHELINESZ;

#ifdef _NVSTATS
	incr_cflush_cntr();
#endif
  }
#endif

  //mfence();
  return;

}

