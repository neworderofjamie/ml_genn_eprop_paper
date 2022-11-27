#!/bin/bash
LINE=$(((($SLURM_ARRAY_TASK_ID - 1)* 4) + $PMI_RANK + 1))
ARGS=`head -${LINE} arguments.txt | tail -1`

python -u classifier.py --train $ARGS
python -u classifier.py $ARGS
