import os
import argparse
import json 

"""
Simple script takes in the current simulation parameters from the config.JSON file and a unique jobID. 
These are used to locate the appropriate macro to run, and also the correct save location for the 
resulting MC.
"""

parser = argparse.ArgumentParser()
parser.add_argument("jobID", type=str, help = "jobID of simulations ")
args = parser.parse_args()

# find current parameters from the config.JSON file
with open("../config.JSON", "r") as config_file:
    config = json.load(config_file)
params  = config["CUR_PARAMS"]
isotope = config["ISOTOPE_USE"] 
num_evs = config["NUM_EVENTS"]
print(num_evs, type(num_evs))
# the command to invoke the RAT simulations
command = f"rat -o /data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/MC/{isotope}/{params}/{args.jobID}.root -n 300823 -N {num_evs} ../macros/{isotope}/{params}.mac"
os.system(command)