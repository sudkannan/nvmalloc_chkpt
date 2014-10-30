#!/bin/bash

set -e

# measure performance
for ((i=0; i<10; i++))
do
	time ./bitmerge `echo 2^22 | bc`
done

## test for every input sizes
#for ((n=1; n<=24; n++))
#do
#	time ./bitmerge `echo 2^$n | bc`
#done
