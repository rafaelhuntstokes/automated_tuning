import numpy as np
import string
import os
import htcondor
import json

def suggest_parameters(isotope):
    t1 = np.random.random(size = 1) * 5
    t2 = (np.random.random(size = 1) + 10) * 1.5
    A1 = np.random.random(size = 1)
    # t1 = [1]
    # t2 = [2]
    # A1 = np.array([0.5])
    A2 = 1 - A1
    fname = f"{round(t1[0], 3)}_{round(t2[0], 3)}_{round(A1[0], 3)}_{round(A2[0], 3)}"
    print(fname)
    
    # load json config file to work out what isotope we are tuning
    
    # create the output directories for the MC files
    if os.path.isdir(f"MC/{isotope}/{fname}"):
        pass
    else:
        os.makedirs(f"MC/{isotope}/{fname}")
    if os.path.isdir(f"macros/{isotope}"):
        pass
    else:
        os.makedirs(f"macros/{isotope}")
    
    # create the template macro
    with open("template_macro.mac", "r") as f:
        rawTextMacro = string.Template(f.read())
    outTextMacro = rawTextMacro.substitute(T1 = round(t1[0], 3), T2 = round(t2[0], 3), A1 = round(A1[0], 3), A2 = round(A2[0], 3))
    with open(f"macros/{isotope}/{fname}.mac", "w") as f:
        f.write(outTextMacro)

    return f"{round(t1[0], 3)}_{round(t2[0], 3)}_{round(A1[0], 3)}_{round(A2[0], 3)}"

def write_subdag(params, NUM_SIMS):
    """
    Depending on the parameters suggested, write the simulation layer of the dag
    workflow.
    """
    from htcondor import dags

    def create_vars_dict(NUM_SIMS):
        """
        Create a list of dictionaries, where each dictionary contains the cmd inputs
        for the subdag executables (ie the output file location of RATDS files and
        the RATMACRO name).
        """
        
        vars = []
        for ijob in range(NUM_SIMS):
            var = dict(OUTFNAME = f"MC/{params[0]}_{params[1]}_{params[2]}_{params[3]}/{ijob}.root",
                       MACNAME  = f"macros/{params[0]}_{params[1]}_{params[2]}_{params[3]}.mac",
                       NUMJOB   = f"{ijob}")
            vars.append(var)
        
        return vars
    
    def create_description():
        """
        Create the submit file description for a single RAT macro submission job.
        """

        description = htcondor.Submit(
            executable = "template.sh",
            arguments  = "$(OUTFNAME) $(MACNAME)",
            output     = "output/$(MACNAME)_$(NUMJOB).output",
            error      = "error/$(MACNAME)_$(NUMJOB).output",
            log        = "log/$(MACNAME)_$(NUMJOB).output"
        )

        return description
    
    subdag = dags.DAG()

    # create the layer of RAT macro jobs being submitted
    rat_description = create_description()
    rat_variables   = create_vars_dict(NUM_SIMS)

    rat_layer = subdag.layer(
        name               = "ratSim",
        submit_description = rat_description,
        vars               = rat_variables
    )

    print(subdag.describe())

    # write the dagfile
    subdag_file = dags.write_dag(subdag, "./ratsims", "ratdag.dag")

# make sh file executable
# os.chmod(f"template.sh", 0o0777)


# update the config.JSON file with the current suggested parameters
with open("config.JSON", "r") as config_file:
    config = json.load(config_file)
ISOTOPE = config["ISOTOPE_USE"]
params = suggest_parameters(ISOTOPE)

config["CUR_PARAMS"] = params

with open("config.JSON", "w") as config_file:
    json.dump(config, config_file, indent = 4)

# update the suggested parameters in the JSON config file (to be read by extractResiduals)
# write_subdag(params, 10)