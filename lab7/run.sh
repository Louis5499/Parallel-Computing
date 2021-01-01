#!/bin/bash
#SBATCH -p pp20
#SBATCH -o horovod.out.%j     # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1
#SBATCH -n 8

export PYTHONPATH=/home/pp20/share/.packages/:$PYTHONPATH
export PATH=/home/pp20/share/.packages/bin:$PATH
horovodrun -np 8 python tf2_mnist_horovod.py
