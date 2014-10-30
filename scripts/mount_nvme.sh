#mkfs.ext4 /dev/nvme0n1
#mkdir /tmp/nvme
#mount /dev/nvme0n1 /mnt/pvm


#!/bin/bash -x
   #

#script to create and mount a pvm
#requires size as input


if [[ x$1 == x ]];
   then
      echo You have specify correct pvm size in MB
      exit 1
   fi



 mkdir /mnt/pvm; chmod 777 /mnt/pvm

 sudo mount -t tmpfs -o size=$1M tmpfs /mnt/pvm


