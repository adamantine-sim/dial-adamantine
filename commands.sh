#!/bin/sh

echo "running adamantine in the container"
pwd
cd /home/volume
ls
rm *vtu
mpirun -n $1 /home/adamantine/bin/adamantine --input-file=input.info