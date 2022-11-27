#!/bin/bash
ARGS=`head -${SLURM_ARRAY_TASK_ID} arguments.txt | tail -1`

python -u classifier.py --train $ARGS
python -u classifier.py $ARGS
