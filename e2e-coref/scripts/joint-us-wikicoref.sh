#!/bin/bash

#SBATCH --job-name=joint-us-wikicoref
#SBATCH --ntasks=1
#SBATCH --output=joint-us-wikicoref.out
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB

source activate e2e-coref
GPU=0 python train.py joint-us-wikicoref