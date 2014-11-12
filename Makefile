LOGGING=/usr/lib64/logging
INCLUDE=$(NVMALLOC_HOME)
src_path=$(NVMALLOC_HOME)
LIB_PATH := $(NVMALLOC_HOME)
BENCH:= $(NVMALLOC_HOME)/compare_bench
CMU_MALLOC:=$(NVMALLOC_HOME)/compare_bench/cmu_nvram/nvmalloc


LDFLAGS=-ldl

# See source code comments to avoid memory leaks when enabling MALLOC_MAG.
#CPPFLAGS := -DMALLOC_PRODUCTION -DMALLOC_MAG
CPPFLAGS := -fPIC -I$(INCLUDE) -I$(INCLUDE)/jemalloc -I/usr/include -g #-I$(INCLUDE)/pmem_intel/linux-examples_flex/libpmem -g #-O3
CPPFLAGS:=  $(CPPFLAGS) -lssl -lcrypto -fPIC
CPPFLAGS := $(CPPFLAGS) -DMALLOC_PRODUCTION -fPIC 
CPPFLAGS := $(CPPFLAGS)  -Wno-pointer-arith
CPPFLAGS := $(CPPFLAGS)  -Wno-unused-function
CPPFLAGS := $(CPPFLAGS)  -Wno-unused-variable -fpermissive
CPPFLAGS := $(CPPFLAGS) -cpp 
CPPFLAGS := $(CPPFLAGS) -D_USENVRAM
#CPPFLAGS := $(CPPFLAGS) -D_NVDEBUG
#CPPFLAGS := $(CPPFLAGS) -D_NOCHECKPOINT
#CPPFLAGS := $(CPPFLAGS) -D_USE_CHECKPOINT
#CPPFLAGS := $(CPPFLAGS) -D_ENABLE_RESTART
#CPPFLAGS := $(CPPFLAGS) -D_ENABLE_SWIZZLING
#CPPFLAGS := $(CPPFLAGS) -D_NVRAM_OPTIMIZE

#cache related flags
#CPPFLAGS := $(CPPFLAGS) -D_USE_CACHEFLUSH
#CPPFLAGS := $(CPPFLAGS) -D_USE_HOTPAGE

#allocator usage
#CPPFLAGS := $(CPPFLAGS) -D_USE_JEALOC_PERSISTONLY
#CPPFLAGS:= $(CPPFLAGS) -D_USE_CMU_NVMALLOC
#CPPFLAGS := $(CPPFLAGS) -D_USERANDOM_PROCID
#CPPFLAGS := $(CPPFLAGS) -D_USE_MALLOCLIB

#can be use together or induvidually
#When using transactions, the logging type
#needs to be specified
#CPPFLAGS := $(CPPFLAGS) -D_USE_SHADOWCOPY
#CPPFLAGS := $(CPPFLAGS) -D_USE_TRANSACTION
#CPPFLAGS := $(CPPFLAGS) -D_USE_UNDO_LOG
#CPPFLAGS := $(CPPFLAGS) -D_USE_REDO_LOG
#CPPFLAGS := $(CPPFLAGS) -D_DUMMY_TRANS
#STATS related flags
#CPPFLAGS := $(CPPFLAGS) -D_NVSTATS
#CPPFLAGS := $(CPPFLAGS) -D_FAULT_STATS 
#emulation related flags
CPPFLAGS := $(CPPFLAGS) -D_USE_FAKE_NVMAP
#CPPFLAGS := $(CPPFLAGS) -D_USE_BASIC_MMAP
#CPPFLAGS := $(CPPFLAGS) -D_USEPIN
#checkpoint related flags
#CPPFLAGS := $(CPPFLAGS) -D_ASYNC_RMT_CHKPT
#CPPFLAGS := $(CPPFLAGS) -D_ASYNC_LCL_CHK
#CPPFLAGS := $(CPPFLAGS) -D_RMT_PRECOPY
#CPPFLAGS := $(CPPFLAGS) -D_COMPARE_PAGES 
#CPPFLAGS:= $(CPPFLAGS) -D_ARMCI_CHECKPOINT
#CPPFLAGS:= $(CPPFLAGS) -D_LIBPMEMINTEL

#Flags that needs to be cleaned later
#NVFLAGS:= -cpp -D_NOCHECKPOINT $(NVFLAGS)
#NVFLAGS:= -D_VALIDATE_CHKSM -cpp $(NVFLAGS)
#NVFLAGS:= -cpp $(NVFLAGS) 
#NVFLAGS:= -cpp -D_USESCR $(NVFLAGS) -lscrf
#NVFLAGS:= $(NVFLAGS) -D_GTC_STATS
#NVFLAGS:= $(NVFLAGS) -D_COMPARE_PAGES -lsnappy
#NVFLAGS:= -cpp $(NVFLAGS) -D_SYNTHETIC
#NVFLAGS:= -D_USENVRAM -cpp $(NVFLAGS)
#NVFLAGS:= $(NVFLAGS) -D_NVSTATS
#NVFLAGS:= $(NVFLAGS) -D_USE_FAKE_NVMAP
#NVFLAGS:= $(NVFLAGS) -D_NOCHECKPOINT 
#NVFLAGS:= $(NVFLAGS) -D_ASYNC_RMT_CHKPT
#NVFLAGS:= $(NVFLAGS) -D_RMT_PRECOPY
#NVFLAGS:= $(NVFLAGS) -D_ASYNC_LCL_CHK
#CXX=g++
#CC=gcc
CXX=mpic++
CC=mpicc

GNUFLAG :=  -std=gnu99 -fPIC -fopenmp 
CFLAGS := -g -I$(INCLUDE) -Wall -pipe -fvisibility=hidden \
	  -funroll-loops  -Wno-implicit -Wno-uninitialized \
	  -Wno-unused-function  -fPIC -fopenmp -g #-larmci 

STDFLAGS :=-std=gnu++0x 
CPPFLAGS := $(CPPFLAGS) -I$(LOGGING)/include -I$(LOGGING)/include/include -I$(LOGGING)/include/port \
	    -I$(CMU_MALLOC)/include  -I$(BENCH) -I$(BENCH)/compare_bench/c-hashtable

LIBS= -lpthread -L$(LOGGING)/lib64  -lm -lssl \
       -Wl,-z,defs -lpthread -lm -lcrypto -lpthread \
       -L$(CMU_MALLOC)/lib 
#	   -lpmem \
#		-lnvmalloc #-llogging

all:  SHARED_LIB 
#BENCHMARK

JEMALLOC_OBJS= 	$(src_path)/jemalloc.o $(src_path)/arena.o $(src_path)/atomic.o \
		$(src_path)/base.o $(src_path)/ckh.o $(src_path)/ctl.o $(src_path)/extent.o \
        $(src_path)/hash.o $(src_path)/huge.o $(src_path)/mb.o \
	    $(src_path)/mutex.o $(src_path)/prof.o $(src_path)/quarantine.o \
	    $(src_path)/rtree.o $(src_path)/stats.o $(src_path)/tcache.o \
	    $(src_path)/util.o $(src_path)/tsd.o $(src_path)/chunk.o \
		$(src_path)/bitmap.o $(src_path)/chunk_mmap.o $(src_path)/chunk_dss.o \
		$(src_path)/np_malloc.o $(src_path)/c_io.o #$(src_path)/malloc_hook.o

RBTREE_OBJS= 	$(src_path)/rbtree.o

NVM_OBJS = $(src_path)/util_func.o $(src_path)/cache_flush.o \
		 $(src_path)/hash_maps.o \
	  	 $(src_path)/checkpoint.o $(src_path)/nv_map.o \
		 $(src_path)/nv_transact.o $(src_path)/nv_stats.o\
		 $(src_path)/gtthread_spinlocks.o  \
		 $(src_path)/nv_debug.o \
		 $(src_path)/pin_mapper.o \
		 #$(src_path)/LogMngr.o\
		 # $(src_path)/c_io.o	
		 #$(src_path)/nv_rmtckpt.cc 
		 #$(src_path)/armci_checkpoint.o  \

BENCHMARK_OBJS = $(BENCH)/c-hashtable/hashtable.o $(BENCH)/c-hashtable/tester.o \
		 $(BENCH)/c-hashtable/hashtable_itr.o $(BENCH)/malloc_bench/nvmalloc_bench.o \
		 $(BENCH)/benchmark.o

$(src_path)/c_io.o: $(src_path)/c_io.cc 
	$(CXX) -c $(src_path)/c_io.cc -o $(src_path)/c_io.o $(LIBS) $(CPPFLAGS)

#$(src_path)/c_io.o: $(src_path)/c_io.c
#	$(CC) -c $(src_path)/c_io.c -o $(src_path)/c_io.o $(LIBS) $(CFLAGS)


$(src_path)/nv_map.o: $(src_path)/nv_map.cc 
	$(CXX) -c $(src_path)/nv_map.cc -o $(src_path)/nv_map.o $(LIBS) $(CPPFLAGS) $(STDFLAGS)

$(src_path)/hash_maps.o: $(src_path)/hash_maps.cc 
	$(CXX) -c $(src_path)/hash_maps.cc -o $(src_path)/hash_maps.o $(LIBS) $(CPPFLAGS) $(STDFLAGS)

$(src_path)/rbtree.o: $(src_path)/rbtree.cc
	$(CXX) -c $(src_path)/rbtree.cc -o $(src_path)/rbtree.o $(LIBS) $(CFLAGS)

$(src_path)/LogMngr.o: $(src_path)/LogMngr.cc
	$(CXX) -c $(src_path)/LogMngr.cc -o $(src_path)/LogMngr.o $(LIBS) $(CPPFLAGS) $(STDFLAGS)

$(src_path)/nv_transact.o: $(src_path)/nv_transact.cc
	$(CXX) -c $(src_path)/nv_transact.cc -o $(src_path)/nv_transact.o $(LIBS) $(CPPFLAGS) $(STDFLAGS)

$(src_path)/nv_stats.o: $(src_path)/nv_stats.cc
	$(CXX) -c $(src_path)/nv_stats.cc -o $(src_path)/nv_stats.o $(LIBS) $(CPPFLAGS) $(STDFLAGS)	

$(src_path)/cache_flush.o: $(src_path)/cache_flush.cc
	$(CXX) -c $(src_path)/cache_flush.cc -o $(src_path)/cache_flush.o $(LIBS) $(CPPFLAGS) $(STDFLAGS)	

$(src_path)/nv_debug.o: $(src_path)/nv_debug.cc
	$(CXX) -c $(src_path)/nv_debug.cc -o $(src_path)/nv_debug.o $(LIBS) $(CPPFLAGS) $(STDFLAGS)	

$(src_path)/pin_mapper.o: $(src_path)/pin_mapper.cc
	$(CXX) -c $(src_path)/pin_mapper.cc -o $(src_path)/pin_mapper.o $(LIBS) $(CPPFLAGS) $(STDFLAGS)	

				
OBJLIST= $(RBTREE_OBJS) $(NVM_OBJS)  $(JEMALLOC_OBJS)
SHARED_LIB: $(RBTREE_OBJS) $(JEMALLOC_OBJS) $(NVM_OBJS)
	$(CC) -c $(RBTREE_OBJS) -I$(INCLUDE) $(CFLAGS) 
	$(CC) -c $(JEMALLOC_OBJS) -I$(INCLUDE) $(CFLAGS) $(NVFLAGS)
	$(CXX) -c $(NVM_OBJS) -I$(INCLUDE) $(CPPFLAGS) $(NVFLAGS)  $(LDFLAGS)
	$(CXX) -shared -fPIC -o libnvmchkpt.so $(OBJLIST) $(NVFLAGS) $(LIBS) $(LDFLAGS)
	ar crf  libnvmchkpt.a $(OBJLIST) $(NVFLAGS)  
	#$(CXX) -g varname_commit_test.cc -o varname_commit_test $(OBJLIST) -I$(INCLUDE) $(CPPFLAGS) $(NVFLAGS)  $(LIBS)


BENCHMARK: $(JEMALLOC_OBJS) $(NVM_OBJS) $(BENCHMARK_OBJS)
	$(CXX) -shared -fPIC -o libnvmchkpt.so $(OBJLIST) -I$(INCLUDE) $(CPPFLAGS) $(NVFLAGS)  $(LIBS)  $(LDFLAGS)
	$(CXX)  $(BENCHMARK_OBJS) -o benchmark $(OBJLIST) -I$(INCLUDE) $(CPPFLAGS) $(NVFLAGS)  $(LIBS)

clean:
	rm -f *.o *.so.0 *.so *.so* nv_read_test
	rm -f nvmalloc_bench
	rm -f test_dirtypgcnt test_dirtypgcpy
	rm -f ../*.o
	rm -f $(BENCHMARK_OBJS) "benchmark" 
	rm -rf libnvmchkpt.*

install:
	mkdir -p /usr/lib64/nvmalloc
	mkdir -p /usr/lib64/nvmalloc/lib
	mkdir -p /usr/lib64/nvmalloc/include
	cp libnvmchkpt.so libnvmchkpt.so.1
	sudo cp libnvmchkpt.so /usr/lib64/nvmalloc/lib/
	sudo cp libnvmchkpt.so.* /usr/lib64/nvmalloc/lib/
	sudo cp libnvmchkpt.so.* /usr/lib/
	sudo cp -r *.h /usr/lib64/nvmalloc/include/
	sudo cp -r *.h /usr/include/
	sudo cp -r *.h /usr/local/include/
	sudo cp libnvmchkpt.so.* /usr/local/lib/
	sudo cp libnvmchkpt.so /usr/local/lib/
	sudo cp libnvmchkpt.so.* /usr/lib/
	sudo cp libnvmchkpt.so /usr/lib/
	#sudo cp pmem_intel/linux-examples_flex/libpmem/libpmem.so  /usr/lib/
	#sudo cp pmem_intel/linux-examples_flex/libpmem/libpmem.so  /usr/lib64/

localinstall:
	mkdir -p ~/nvmchkpt/include
	mkdir -p ~/nvmchkpt/lib
	cp libnvmchkpt.so ~/nvmchkpt/lib
	cp *.h ~/nvmchkpt/include


uninstall:
	rm -rf libnvmchkpt.*
	rm -rf /usr/lib64/nvmalloc/lib/libnvmchkpt.so*
	sudo rm -rf /usr/local/lib/libnvmchkpt.so*
	sudo rm -rf /usr/lib/libnvmchkpt.so*
