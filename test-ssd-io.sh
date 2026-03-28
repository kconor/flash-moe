#!/bin/bash 

sudo purge && fio \
  --name=expert_randread \
  --filename=models/Qwen3.5-122B-A10B-4bit/packed_experts/layer_43.bin \
  --rw=randread \
  --bs=5300k \
  --numjobs=1 \
  --iodepth=1 \
  --readonly \
  --runtime=10 \
  --direct=1 \
  --time_based \
  --group_reporting
