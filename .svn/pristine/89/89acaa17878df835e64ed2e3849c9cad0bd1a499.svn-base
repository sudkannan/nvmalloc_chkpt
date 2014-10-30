#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define __NR_NValloc 299

int main(int argc, char *argv[])
{
  syscall(__NR_NValloc, 0);
  return 0;
}
