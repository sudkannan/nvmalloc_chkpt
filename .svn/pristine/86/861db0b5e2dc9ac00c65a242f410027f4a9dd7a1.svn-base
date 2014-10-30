# Makefile
# File: "/home/llpilla/Desktop/cuda_ep/Makefile"
# Created: "Wed, 09 Sep 2009 11:28:19 -0300 (llpilla)"
# Updated: "Wed, 16 Sep 2009 13:49:59 -0300 (llpilla)"

CHKPT=/nethome/skannan9/PROJECTS
BASE =vecAdd
INCLUDE = -m64 -g  -I/opt/cuda-4.0/cuda/include -I$(CHKPT) -I. -I.. -I/home/sudarsun/NVIDIA_GPU_Computing_SDK/C/common/inc -lpthread -L/opt/cuda-4.0/cuda/lib64 -lcuda

CPPFLAGS:= $(CPPFLAGS) -D_USECHKPT
CPPFLAGS:= $(CPPFLAGS) -D_USENVRAM 
CPPFLAGS:= $(CPPFLAGS) -D_USE_PRECOPY
#CPPFLAGS:= $(CPPFLAGS) -D_VALIDATE_CHKSM

FILES = vecAdd.cu $(CHKPT)/timer.c  $(CHKPT)/nvmalloc/util_func.cc $(CHKPT)/checkpoint.cu
LIBS= -lssl -lcrypto -L$(CHKPT)/nvmalloc -lnvmchkpt
EXIT = vecAdd

all:
	/opt/cuda-4.0/cuda/bin/nvcc ${FILES} -o ${EXIT} -I ${INCLUDE} $(CPPFLAGS) $(LIBS) -O2  -arch sm_20

clean:
	rm -rf BurtscherGCG.2.2 $(CHKPT)/checkpoint.o
