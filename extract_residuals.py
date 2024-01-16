import numpy as np 
import rat 
from ROOT import RAT
import argparse
import os 
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
def extractAnalysis(parameters, isotope, FV_CUT, zOffset, ENERGY_LOW, ENERGY_HIGH):
    """
    INPUTS: These inputs define the MC file to load in:

            1 ) emission time parameters STRING
            2 ) iteration name           STRING
            3 ) isotope simulated        STRING
            
            Inputs define cuts to apply to MC:

            1 ) FV_CUT                   FLOAT
            2 ) zOffset                  FLOAT
            3) ENERGY_LOW                FLOAT
            4) ENERGY_HIGH               FLOAT

    OUTPUT: time residuals               ARRAY
    """ 

    # OUTPUTS
    residualsRECON = []
    COUNTER        = 0 
    
    fname = f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/MC/{isotope}/{parameters}/*.root"
    print(fname)
    ds = RAT.DU.DSReader(fname)
        
    for ientry, _ in rat.dsreader(fname):
        # light path calculator and point3D stuff loaded after ratds constructor
        # timeResCalc = rat.utility().GetTimeResidualCalculator()
        PMTCalStatus = RAT.DU.Utility.Get().GetPMTCalStatus()
        light_path = rat.utility().GetLightPathCalculator()
        group_velocity = rat.utility().GetGroupVelocity()
        pmt_info = rat.utility().GetPMTInfo()
        psup_system_id = RAT.DU.Point3D.GetSystemId("innerPMT")
        av_system_id = RAT.DU.Point3D.GetSystemId("av")
        
        # entry = ds.GetEntry(i)
        if ientry.GetEVCount() == 0:
            continue

        #### RECONSTRUCTION INFORMATION EXTRACTED ####
        reconEvent = ientry.GetEV(0)
        
        # did event get reconstructed correctly?
        fit_name = reconEvent.GetDefaultFitName()
        if not reconEvent.FitResultExists(fit_name):
            continue

        vertex = reconEvent.GetFitResult(fit_name).GetVertex(0)
        if (not vertex.ContainsPosition() or
            not vertex.ContainsTime() or
            not vertex.ValidPosition() or
            not vertex.ValidTime() or
            not vertex.ContainsEnergy() or
            not vertex.ValidEnergy()):
            continue
        # print("Reconstruction checks PASSED!")
        # reconstruction valid so get reconstructed position and energy
        reconPosition  = vertex.GetPosition() # returns in PSUP coordinates
        reconEnergy    = vertex.GetEnergy()        
        reconEventTime = vertex.GetTime()

        # apply AV offset to position
        event_point = RAT.DU.Point3D(psup_system_id, reconPosition)
        event_point.SetCoordinateSystem(av_system_id)
        if event_point.Mag() > FV_CUT:
            continue
        # convert back to PSUP coordinates
        event_point.SetCoordinateSystem(psup_system_id)

        # apply energy tagging cuts the same as that in data
        if reconEnergy < ENERGY_LOW or reconEnergy > ENERGY_HIGH:
            continue
        
        # event has passed all the cuts so we can extract the time residuals
        calibratedPMTs = reconEvent.GetCalPMTs()
        pmtCalStatus = rat.utility().GetPMTCalStatus()
        for j in range(calibratedPMTs.GetCount()):
            pmt = calibratedPMTs.GetPMT(j)
            if pmtCalStatus.GetHitStatus(pmt) != 0:
                continue
            
            # residual_recon = timeResCalc.CalcTimeResidual(pmt, reconPosition, reconEventTime, True)
            pmt_point = RAT.DU.Point3D(psup_system_id, pmt_info.GetPosition(pmt.GetID()))
            light_path.CalcByPosition(event_point, pmt_point)
            inner_av_distance = light_path.GetDistInInnerAV()
            av_distance = light_path.GetDistInAV()
            water_distance = light_path.GetDistInWater()
            transit_time = group_velocity.CalcByDistance(inner_av_distance, av_distance, water_distance)
            residual_recon = pmt.GetTime() - transit_time - reconEventTime
            
            residualsRECON.append(residual_recon)
        
        COUNTER += 1
        if COUNTER % 1 == 0:
            print("COMPLETED {} / {}".format(COUNTER, ds.GetEntryCount()))
    
    return residualsRECON

def create_graphics(data, MC, params, isotope, model):
    binning = np.arange(-5, 350, 1)

    plt.figure()
    plt.hist(data, bins = binning, density = True, histtype = "step", label = "Data")
    plt.hist(MC, bins = binning, density = True, histtype = "step", label = "MC")
    plt.title(f"{isotope} | {params}")
    plt.xlabel("Time Residual (ns)")
    plt.ylabel("Normalised Counts per 1 ns Bin")
    plt.legend()
    plt.xlim((-5, 100))
    plt.savefig(f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/plots/{isotope}/{params}_{model}.png")
    plt.close()

    # log plot of the tails
    binning = np.arange(-5, 350, 1)
    plt.figure()
    plt.yscale("log")
    plt.hist(data, bins = binning, density = True, histtype = "step", label = "Data")
    plt.hist(MC, bins = binning, density = True, histtype = "step", label = "MC")
    plt.title(f"{isotope} | {params}")
    plt.xlabel("Time Residual (ns)")
    plt.ylabel("Normalised Counts per 1 ns Bin")
    plt.legend()
    plt.savefig(f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/plots/{isotope}/{params}_{model}_TAILS.png")
    plt.close()
    
    # cost vs iteration graph
    plt.figure()
    cost = config["COST"]
    print(cost)
    plt.plot(cost)
    plt.title("Cost Function per Iteration")
    plt.xlabel("Iteration #")
    plt.ylabel(r"$\chi ^2$")
    plt.savefig(f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/plots/{isotope}/cost_vs_iter_{model}.pdf")
    plt.close()

if __name__ == "__main__":

    """
    Find the current timing parameters used to simulate in this loop from the config.JSON file.
    Use this to load the respective MC .root files.

    Also find the AV OFFSET, FV_CUTS, ENERGY CUTS used for the given ISOTOPE from the config.JSON file.

    Extract the time residuals from the MC.

    Calculate the chi2 vs the respective data residuals.

    Save the residuals, chi2, and a plot of data vs MC residuals for this iteration.
    """
    
    # open up the json config file to read in the settings for evaluating time residuals
    with open("config.JSON", "r") as config_file:
        config = json.load(config_file)
    
    # extract the necessary parameters from config file
    ISOTOPE           = config["ISOTOPE_USE"]
    TIMING_PARAMETERS = config["CUR_PARAMS"]
    AV_OFFSET         = config["AV_OFFSET"]
    FV_CUT            = config["FV_CUT"]
    E_LOW             = config["ISOTOPE_E"][f"{ISOTOPE}"]["E_LOW"]
    E_HIGH            = config["ISOTOPE_E"][f"{ISOTOPE}"]["E_HIGH"]
    DOMAIN_LOW        = config["DOMAIN_LOW"] 
    DOMAIN_HIGH       = config["DOMAIN_HIGH"]
    GLOBAL_BEST       = config["GLOBAL_BEST"]["CHI2"]
    MODEL             = config["TUNING_MODEL"]

    # use these parameters to extract the residuals with the right cuts and from the right MC
    residuals  = extractAnalysis(TIMING_PARAMETERS, ISOTOPE, FV_CUT, AV_OFFSET, E_LOW, E_HIGH)

    np.save(f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/residuals/{ISOTOPE}/{TIMING_PARAMETERS}.npy", residuals)
    
    # COMPUTE THE SQUARED DIFFERENCES BETWEEN THIS AND THE DATA
    # actually due to changing normalisation conditions, need to bin over the entire range
    binning = np.arange(-5, 350, 1)
    data = np.load(f"/data/snoplus3/hunt-stokes/tune_cleaning/detector_data/{ISOTOPE}_{FV_CUT}mmFV_{E_LOW}MeV_{E_HIGH}MeV_data_residuals3.npy", allow_pickle=True)
    print(len(data))
    data = np.concatenate(data)
    data_hist_counts, data_hist_bin_edges = np.histogram(data, bins = binning, density = True)
    model_hist_counts, model_hist_bin_edges = np.histogram(residuals, bins = binning, density = True)

    # find the bin idx from -5 ns --> 40 ns to compute the chi2
    binIdxLow = np.where(binning == int(DOMAIN_LOW))[0][0]
    binIdxHigh = np.where(binning == int(DOMAIN_HIGH))[0][0]

    # compute diffs of each bin counts 
    diffs = np.sum(( data_hist_counts[binIdxLow:binIdxHigh] - model_hist_counts[binIdxLow:binIdxHigh] ) **2 )

    # update the COST FUNCTION value for this sample
    cost = config["COST"]
    cost.append(diffs)

    # save this number
    np.save(f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/chi2/{ISOTOPE}/{TIMING_PARAMETERS}.npy", diffs)

    # check if the residual is better than the current global best --> if yes, update global best params and cost
    if diffs < GLOBAL_BEST:
        if MODEL == "doubleExponential":
            config["GLOBAL_BEST"]["T1"]   = config["MEASURED_POINTS"]["T1"][-1]
            config["GLOBAL_BEST"]["T2"]   = config["MEASURED_POINTS"]["T2"][-1]
            config["GLOBAL_BEST"]["A1"]   = config["MEASURED_POINTS"]["A1"][-1]
            config["GLOBAL_BEST"]["A2"]   = config["MEASURED_POINTS"]["A2"][-1]
            config["GLOBAL_BEST"]["CHI2"] = diffs
        if MODEL == "tripleExponential":
            config["GLOBAL_BEST"]["T1"]   = config["MEASURED_POINTS"]["T1"][-1]
            config["GLOBAL_BEST"]["T2"]   = config["MEASURED_POINTS"]["T2"][-1]
            config["GLOBAL_BEST"]["T3"]   = config["MEASURED_POINTS"]["T3"][-1]
            config["GLOBAL_BEST"]["A1"]   = config["MEASURED_POINTS"]["A1"][-1]
            config["GLOBAL_BEST"]["A2"]   = config["MEASURED_POINTS"]["A2"][-1]
            config["GLOBAL_BEST"]["A3"]   = config["MEASURED_POINTS"]["A3"][-1]
            config["GLOBAL_BEST"]["CHI2"] = diffs
        if MODEL == "quadrupleExponential":
            config["GLOBAL_BEST"]["T1"]       = config["MEASURED_POINTS"]["T1"][-1]
            config["GLOBAL_BEST"]["T2"]       = config["MEASURED_POINTS"]["T2"][-1]
            config["GLOBAL_BEST"]["T3"]       = config["MEASURED_POINTS"]["T3"][-1]
            config["GLOBAL_BEST"]["T4"]       = config["MEASURED_POINTS"]["T4"][-1]
            config["GLOBAL_BEST"]["TR"]       = config["MEASURED_POINTS"]["TR"][-1]
            config["GLOBAL_BEST"]["THETA1"]   = config["MEASURED_POINTS"]["THETA1"][-1]
            config["GLOBAL_BEST"]["THETA2"]   = config["MEASURED_POINTS"]["THETA2"][-1]
            config["GLOBAL_BEST"]["THETA3"]   = config["MEASURED_POINTS"]["THETA3"][-1]
            config["GLOBAL_BEST"]["CHI2"] = diffs
        
    with open("config.JSON", "w") as f:
        json.dump(config, f, indent = 4)
    
    create_graphics(data, residuals, TIMING_PARAMETERS, ISOTOPE, MODEL)