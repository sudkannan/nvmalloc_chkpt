.DEFAULT_GOAL := all




CC = gcc
CFLAGS =
COPTFLAGS = -O3 -g

CXX = g++ #icpc
CXXOPTFLAGS = $(COPTFLAGS)
CXXFLAGS =

CHKPT=/nethome/skannan9/PROJECTS
NVINCLUDE:= -m64 -g  -I/opt/cuda-4.0/cuda/include -I$(CHKPT) -I. -I.. -I/home/sudarsun/NVIDIA_GPU_Computing_SDK/C/common/inc
NVLIBS= -lssl -lcrypto -lpthread -L/opt/cuda-4.0/cuda/lib64 -lcuda  -L$(CHKPT)/nvmalloc -lnvmchkpt 
NVFLAGS:= $(NVINCLUDE)  $(NVFLAGS) -D_USECHKPT 
NVFLAGS:= $(NVFLAGS) -D_USENVRAM
NVGPUSRC:=$(CHKPT)/checkpoint.cu
NVGPUOBJ:=$(CHKPT)/checkpoint.o


NVCC = $(shell which nvcc)
NVCFLAGS = -arch=sm_20 --ptxas-options=-v
NVCOPTFLAGS = -O3
PATH_CUDART = $(shell dirname $(NVCC))/../lib64
LDFLAGS_CUDA = -L$(PATH_CUDART) -Wl,-rpath $(PATH_CUDART) -lcuda -lcudart

HDRS_C = timer.h
SRCS_C = $(HDRS_C:.h=.c)
OBJS_C = $(SRCS_C:.c=.o)

HDRS_CC = bitmerge.h bitmerge-seq.h #bitmerge-cilk.h
SRCS_CC = $(HDRS_CC:.h=.cc)
OBJS_CC = $(SRCS_CC:.cc=.o)

HDRS_CUDA = bitmerge-cuda.h
SRCS_CUDA = $(HDRS_CUDA:.h=.cu)
OBJS_CUDA = $(SRCS_CUDA:.cu=.o)

all: bitmerge
	@echo "=== Success! ==="

bitmerge: $(OBJS_C) $(OBJS_CC) $(OBJS_CUDA) $(NVGPUOBJ)
#bitmerge: $(OBJS_C) $(OBJS_CUDA)
	$(CXX) $(CXXOPTFLAGS) $(NVFLAGS)  -o $@ $^ $(LDFLAGS_CUDA) $(NVLIBS)

# Valid in the instructor's kit only
soln: bitmerge--soln$(SID)
bitmerge--soln$(SID): $(OBJS_C) $(OBJS_CC) bitmerge-cuda--soln$(SID).o
	$(CXX) $(CXXOPTFLAGS)  $(NVFLAGS) -o $@ $^ $(LDFLAGS_CUDA) $(NVLIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(CXXOPTFLAGS) $(NVFLAGS) -o $@ -c $<

%.o: %.c
	$(CC) $(CFLAGS) $(COPTFLAGS) $(NVFLAGS) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(NVCFLAGS) $(NVFLAGS) $(NVCOPTFLAGS) -o $@ -c $<

clean:
	rm -f core *.o *~ bitmerge

snap:
	@if ! test -d snapshots ; then \
	  mkdir snapshots ; \
	  LASTSNAP=-1 ; \
	else \
	  LASTSNAP=$$(find ./snapshots -type d | awk -F / '{ print $$NF; }' | grep -v snapshots | sort -n | tail -1) ; \
	  if test -z "$${LASTSNAP}" ; then LASTSNAP=-1 ; fi ; \
	fi ; \
	NEXTSNAP=$$(($${LASTSNAP} + 1)) ; \
	mkdir snapshots/$${NEXTSNAP} ; \
	echo "=== Taking snapshot: snapshots/$${NEXTSNAP} ===" ; \
	cp $(HDRS_C) $(SRCS_C) $(HDRS_CC) $(SRCS_CC) $(HDRS_CUDA) $(SRCS_CUDA) \
	    snapshots/$${NEXTSNAP}/.

snap-pack:
	@if test -d snapshots ; then \
	  tar cvf - snapshots/* | bzip2 -9c > snapshots.tar.bz2 ; \
	fi

run: bitmerge
	../submit-and-wait.sh bitmerge.pbs

cleanoutput:
	\rm -f *.[eo][0-9][0-9][0-9][0-9][0-9]

tag:
	ctags --langmap=c:+.cu *.c *.cc *.cu *.h

# eof
