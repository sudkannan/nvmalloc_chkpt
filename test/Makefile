CXX=g++ #/net/hu14/skannan9/mvapich2-1.8-r5471/install_libs/bin/mpicxx
CC=gcc #/net/hu14/skannan9/mvapich2-1.8-r5471/install_libs/bin/mpicc
GNUFLAG :=  -std=gnu99 

INCLUDE=$(NVMALLOC_HOME)

DEFINES := -DMALLOC_PRODUCTION -lstdc++


CFLAGS := -g -I$(INCLUDE) -Wall -pipe -fvisibility=hidden \
      -funroll-loops  -Wno-implicit -Wno-uninitialized \
      -Wno-unused-function -Wno-unused-variable -lm -lssl  -fPIC

#LIBS= -ljemalloc -lpthread
LIBS= -lpthread -lnvmchkpt

all:  nv_read_test nv_read_test_alloc test_dirtypgcpy nvmalloc_bench varname_alloc_test varname_read_test varname_commit_test \
      nvmap_bench \
      nvmalloc_non_persist_bench nvmalloc_cc_test nvmalloc_c_test nv_restart_test test_del \
	  load_file	read_file
	 #nv_remote_chkpt_test nv_rmtckpt_nodirtchk nv_test_spinlock nv_chckpt_test

nv_read_test:  nv_read_test.cc
	$(CXX) nv_read_test.cc -o nv_read_test  $(CFLAGS) $(LIBS) $(DEFINES)

nv_read_test_alloc:  nv_read_test_alloc.cc
	$(CXX) nv_read_test_alloc.cc -o nv_read_test_alloc  $(CFLAGS) $(LIBS) $(DEFINES)

nvmalloc_bench: nvmalloc_bench.cc
	$(CXX) nvmalloc_bench.cc -o nvmalloc_bench  $(CFLAGS) $(LIBS) -I$(INCLUDE) $(DEFINES)

test_dirtypgcpy: test_dirtypgcpy.c
	$(CXX) test_dirtypgcpy.c -o test_dirtypgcpy  $(CFLAGS) $(LIBS) $(DEFINES)

varname_alloc_test: varname_alloc_test.cc
	$(CXX)	varname_alloc_test.cc -o varname_alloc_test  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

varname_read_test: varname_read_test.cc
	$(CXX)	varname_read_test.cc -o varname_read_test  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

varname_commit_test: varname_commit_test.cc
	$(CXX)	varname_commit_test.cc -o varname_commit_test  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)


nv_chckpt_test: nv_chckpt_test.cc
	$(CXX) nv_chckpt_test.cc -o nv_chckpt_test  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

nv_restart_test:  nv_restart_test.cc
	$(CXX) nv_restart_test.cc -o nv_restart_test  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

nvmap_bench: nvmap_bench.cc
	$(CXX) nvmap_bench.cc -o nvmap_bench  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

nvmalloc_cc_test: nvmalloc_cc_test.cc
	$(CXX) nvmalloc_cc_test.cc -o nvmalloc_cc_test  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

nvmalloc_c_test: nvmalloc_c_test.c
	$(CC) nvmalloc_c_test.c -o nvmalloc_c_test  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)


nvmalloc_non_persist_bench: nvmalloc_non_persist_bench.cc
	$(CXX) nvmalloc_non_persist_bench.cc -o nvmalloc_non_persist_bench  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

test_del: test_del.cc
	$(CXX) test_del.cc -o test_del  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

load_file: load_file.cc
	$(CC) load_file.cc -o load_file  $(CFLAGS) $(LIBS) -I$(INCLUDE) $(DEFINES)

read_file: read_file.cc
	$(CC) read_file.cc -o read_file  $(CFLAGS) $(LIBS) -I$(INCLUDE) $(DEFINES)

#nv_remote_chkpt_test:  nv_remote_chkpt_test.cc
#	$(CXX) nv_remote_chkpt_test.cc -o nv_remote_chkpt_test  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

#nv_rmtckpt_nodirtchk:  nv_rmtckpt_nodirtchk.cc
#	$(CXX) nv_rmtckpt_nodirtchk.cc -o nv_rmtckpt_nodirtchk  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

nv_test_spinlock:  nv_test_spinlock.cc 
	$(CXX) nv_test_spinlock.cc -o nv_test_spinlock  $(CFLAGS) $(LIBS) -I$(INCLUDE)/jemalloc $(DEFINES)

clean:
	rm -f *.o *.so.0 *.so nv_read_test
	rm -f nvmalloc_bench nv_read_test_alloc nvmap_bench
	rm -f test_dirtypgcnt test_dirtypgcpy varname_alloc_test
	rm -f varname_read_test nv_chckpt_test nv_restart_test
	rm -f nv_remote_chkpt_test nv_rmtckpt_nodirtchk
	rm -f nvmalloc_non_persist_bench
	rm -f nvmalloc_cc_test nvmalloc_c_test hello_world
