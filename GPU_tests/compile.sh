 
#/opt/cuda-4.0/cuda/bin/nvcc -m64  -I/opt/cuda-4.0/cuda/include -I. -I.. -Icommon/inc  -L/opt/cuda-4.0/cuda/lib64 -lcuda timer.c checkpoint.cu  vecOrig.cu -o vecOrig
#/opt/cuda-4.0/cuda/bin/nvcc -arch sm_11 -m64  -I/opt/cuda-4.0/cuda/include -I. -I.. -Icommon/inc  -L/opt/cuda-4.0/cuda/lib64 -lcuda timer.c checkpoint.cu  $1.cu -o $1
/opt/cuda-4.0/cuda/bin/nvcc -m64 -arch sm_20   -I/opt/cuda-4.0/cuda/include -I. -I.. -Icommon/inc  -L/opt/cuda-4.0/cuda/lib64 -lcuda timer.c checkpoint.cu  $1.cu -o $1
