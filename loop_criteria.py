import json
import sys
import shutil
with open("config.JSON", "r") as f:
    config = json.load(f)

num_loops = config["CUR_LOOP"]
max_loops = config["MAX_LOOP"]
with open("config.JSON", "w") as f:
    if num_loops < max_loops:
        

        config["CUR_LOOP"] = num_loops + 1
        json.dump(config, f, indent= 4)

        # copy the suggest log from the previous iteration to a unique file 
        # this avoids it being overwritten by the next loop of the dag!
        shutil.copyfile("/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/suggest.out", f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/suggest_{num_loops}.out")
        shutil.copyfile("/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/residuals.out", f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/residuals_{num_loops}.out")
        shutil.copyfile("/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/suggest.err", f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/suggest_{num_loops}.err")
        shutil.copyfile("/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/residuals.error", f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/logs/residuals_{num_loops}.error")
        sys.exit(1)
    else:
        # we have reached max iters for this tuning model --> proceed to the next model and reset CUR_LOOP
        config["CUR_LOOP"] = 1
        curr_model = config["TUNING_MODEL"]
        if curr_model == "doubleExponential":
            # reset the measured points array and cost (the GLOBAL BEST value is saved)
            config["MEASURED_POINTS"]["T1"] = []
            config["MEASURED_POINTS"]["T2"] = []
            config["MEASURED_POINTS"]["T3"] = []
            config["MEASURED_POINTS"]["T4"] = []
            config["MEASURED_POINTS"]["A1"] = []
            config["MEASURED_POINTS"]["A2"] = []
            config["MEASURED_POINTS"]["A3"] = []
            config["MEASURED_POINTS"]["A4"] = []
            config["COST"]                  = []
            config["TUNING_MODEL"] = "tripleExponential"
        if curr_model == "tripleExponential":
            # reset the measured points array and cost (the GLOBAL BEST value is saved)
            config["MEASURED_POINTS"]["T1"] = []
            config["MEASURED_POINTS"]["T2"] = []
            config["MEASURED_POINTS"]["T3"] = []
            config["MEASURED_POINTS"]["T4"] = []
            config["MEASURED_POINTS"]["A1"] = []
            config["MEASURED_POINTS"]["A2"] = []
            config["MEASURED_POINTS"]["A3"] = []
            config["MEASURED_POINTS"]["A4"] = []
            config["COST"]                  = []
            config["TUNING_MODEL"] = "quadrupleExponential"
        if curr_model == "quadrupleExponential":
            # we have reached the end of the quadruple exponential tuning and we don't want to continue from here
            print("Finished tuning!")
        json.dump(config, f, indent = 4)
        sys.exit(0)

