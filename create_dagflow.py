import htcondor
from htcondor import dags
import json
import string

"""
Script creates the DAG workflow for the 'inner' DAG. The outer DAG handles recursion only.

LAYER 1: suggestParameters.py --> suggest timing parameters
                              --> create and save rat macro for simulating with those parameters
                              --> update config.JSON file with these parameters
        
        This layer does not take any input arguments.

LAYER 2: runSimulations.py    --> INPUT: the jobID (e.g 1 --> N simulations run)
                              --> create wide layer of N python scripts
                              --> each script loads the config.JSON file to work out which macro to run
                              --> each script takes a jobID as an input argument (determine output file)

LAYER 3: extractResiduals.py  --> load the MC for the current parameters (obtained from config.JSON)
                              --> extract time residuals from the MC
                              --> find the chi2 difference of residuals compared to data
                              --> save the plot of time residuals MC vs data, residuals themselves and chi2
"""

def run_simulations_vars():
    """
    Create a list of dictionaries. Each dictionary contains the input args for a RAT 
    simulation --> in this case, just the jobID number to allow saving unique output files.
    
    Find the number of rat simulations from the config.JSON file.
    """
    with open("config.JSON") as config_file:
        config    = json.load(config_file)
        num_jobs  = config["NUM_SIMS"]
    vars = []
    for ijob in range(num_jobs):
        var = dict(SIM_NUM = ijob)
        vars.append(var)
    return vars

suggest_params_description = htcondor.Submit(
    executable = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/suggest_constants.sh',
    output     = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/suggest.out',
    error      = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/suggest.err',
    log        = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/suggest.log'
)

run_simulations_description = htcondor.Submit(
    executable = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/run_simulations.sh',
    arguments  = '$(SIM_NUM)',
    output     = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/$(SIM_NUM)_simulate.out',
    error      = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/$(SIM_NUM)_simulate.err',
    log        = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/$(SIM_NUM)_simulate.log'
)

extract_residuals_description = htcondor.Submit(
    executable = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/extract_residuals.sh',
    output     = "/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/residuals.out",
    error      = "/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/residuals.error",
    log        = "/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/residuals.log"
)

simulation_vars = run_simulations_vars()

# create the nested subdag and add each layer
dag = dags.DAG()
suggest_params_layer = dag.layer(name = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/dag/suggest',
                                 submit_description = suggest_params_description,
                                 )
simulate_layer = suggest_params_layer.child_layer(name = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/dag/simulate',
                                                  submit_description = run_simulations_description,
                                                  vars = simulation_vars)
residuals_layer = simulate_layer.child_layer(name = '/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/dag/residuals',
                                             submit_description = extract_residuals_description,
                                            )


dag_file = dags.write_dag(dag, "./dag", "algo.dag")

# create the outermost dag which controls the looping behaviour
# can't get the subdag API to work so just using a template string to fill in the number of retries
with open("config.JSON", "r") as config_file:
    config = json.load(config_file)
num_loops = config["MAX_LOOP"]

with open("loop_template.dag", "r") as dag_file:
    template = string.Template(dag_file.read())
output_text = template.substitute(NUM_RETRIES = num_loops)

with open("./dag/loop.dag", "w") as dag_file:
    dag_file.write(output_text) 