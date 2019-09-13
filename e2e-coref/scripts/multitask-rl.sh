#!/bin/bash

#SBATCH --job-name=multitask-rl
#SBATCH --ntasks=1
#SBATCH --output=multitask-rl.out
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB

source activate e2e-coref
GPU=0 python train.py multitask-rl