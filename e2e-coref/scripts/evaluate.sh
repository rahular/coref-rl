#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --output=evaluate.out
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB

source activate e2e-coref

# Build custom kernels.
# TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
# TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Linux (pip)
# g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# rm ./logs/final-wikicoref/*
# cp ./logs/text-us-wikicoref/model.max.ckpt.* ./logs/final-wikicoref/
# GPU=3 python evaluate.py final-wikicoref
# echo "==== end: text-us-wikicoref ===="

# rm ./logs/final-wikicoref/*
# cp ./logs/kg-us-wikicoref/model.max.ckpt.* ./logs/final-wikicoref/
# GPU=1 python evaluate.py final-wikicoref
# echo "==== end: kg-us-wikicoref ===="

# rm ./logs/final-wikicoref/*
# cp ./logs/joint-us-wikicoref/model.max.ckpt.* ./logs/final-wikicoref/
# GPU=1 python evaluate.py final-wikicoref
# echo "==== end: joint-us-wikicoref ===="

# rm ./logs/final/*
# cp ./logs/text-us/model.max.ckpt.* ./logs/final/
# GPU=1 python evaluate.py final
# echo "==== end: text-us ===="

# rm ./logs/final/*
# cp ./logs/kg-us/model.max.ckpt.* ./logs/final/
# GPU=3 python evaluate.py final
# echo "==== end: kg-us ===="

# rm ./logs/final/*
# cp ./logs/joint-us/model.max.ckpt.* ./logs/final/
# GPU=1 python evaluate.py final
# echo "==== end: joint-us ===="

# rm ./logs/final/*
# cp ./logs/multitask-rl/model.max.ckpt.* ./logs/final/
# GPU=3 python evaluate.py final
# echo "==== end: multitask-rl ===="

# rm ./logs/final-wikicoref/*
# cp ./logs/multitask-rl-wikicoref/model.max.ckpt.* ./logs/final-wikicoref/
# GPU=3 python evaluate.py final-wikicoref
# echo "==== end: multitask-rl-wikicoref ===="

echo "==== start: final-ontonotes ===="
GPU=3 python evaluate.py final
echo "==== end: final-ontonotes ===="

echo "==== start: final-wikicoref ===="
GPU=3 python evaluate.py final-wikicoref
echo "==== end: final-wikicoref ===="