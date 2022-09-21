#!/bin/bash
# Run decode.py with GPU

cmd="utils/queue.pl --mem 8G --gpu 1 --config conf/gpu.conf"

${cmd} decode.log \
  bash -c 'export CUDA_VISIBLE_DEVICES=$(free-gpu); /export/c07/sli218/miniconda3/envs/OVAD/bin/python3.8 ./decode.py'
