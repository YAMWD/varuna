#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8

python -m varuna.run_varuna --machine_list list --no_morphing --gpus_per_node 1 --batch_size 2 --nstages 1 --chunk_size 1 --code_dir . main.py data -e -a 'efficientnet-b0' --pretrained --varuna --lr 0.001 --epochs 1 --cpu
