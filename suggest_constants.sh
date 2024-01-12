#!/bin/bash
source /data/snoplus3/hunt-stokes/automated_tuning/auto_env/bin/activate
source /home/hunt-stokes/rat4_env.sh
cd /data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2

python3 baysOpt.py