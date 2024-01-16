#!/bin/bash

source /home/hunt-stokes/rat4_env.sh
cd /data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/ratlogs

python3 ../run_simulation.py $1
# rm rat*.log