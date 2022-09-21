#!/bin/bash
# Run train.py with GPU

cmd="utils/queue.pl --mem 10G --gpu 1 --config conf/gpu.conf"

${cmd} train.log \
  bash -c 'export CUDA_VISIBLE_DEVICES=$(free-gpu); /export/c07/sli218/miniconda3/envs/OVAD/bin/python3.8 ./train.py'
