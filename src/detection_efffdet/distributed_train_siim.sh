#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=1357 train_siim.py "$@"

