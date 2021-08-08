#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=1234 warmup_ext_dataset.py "$@"

