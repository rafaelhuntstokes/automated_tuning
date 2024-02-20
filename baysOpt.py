import json
import numpy as np
import string
import os
from scipy import stats
def setup_predictions_double():
    T1 = np.arange(0.1, 10, 1)
    T2 = np.arange(5, 30, 1)
    A1 = np.arange(0.6, 1.0, 0.1)

    predicted_points = np.zeros((len(T1)*len(T2)*len(A1), 4))
    pt_num           = 0
    for iT1 in T1:
        for iT2 in T2:
            for iA1 in A1:
                predicted_points[pt_num, 0] = iT1
                predicted_points[pt_num, 1] = iT2
                predicted_points[pt_num, 2] = iA1
                predicted_points[pt_num, 3] = 1 - iA1
                pt_num +=1

    return predicted_points

def setup_predictions_triple(T1, T2, A1, A2):
    """
    Scan over T3 values, keep A1:A2 fixed.
    """

    T3 = np.arange(50, 300, 10)
    A3 = np.arange(0, 0.15, 0.01)

    sigma            = 1 + A2/A1
    predicted_points = np.zeros((len(T3)*len(A3), 6))
    pt_num           = 0 
    for iT3 in T3:
        for iA3 in A3:
            delta1       = (iA3 / sigma) * 1       # subtract this much from A1
            delta2       = (iA3 / sigma) * A2 / A1 # subtract this much from A2
            A1_corrected = A1 - delta1
            A2_corrected = A2 - delta2

            predicted_points[pt_num, 0] = T1
            predicted_points[pt_num, 1] = T2
            predicted_points[pt_num, 2] = iT3
            predicted_points[pt_num, 3] = A1_corrected
            predicted_points[pt_num, 4] = A2_corrected
            predicted_points[pt_num, 5] = iA3
            pt_num += 1

    return predicted_points
            # 

def setup_predictions_quadruple(T1, T2, T3, A1, A2, A3):
    """
    Scan over T4 values, keeping A1:A2:A3 ratio fixed.
    """

    T4 = np.arange(350, 500, 10)
    A4 = np.arange(0, 0.05, 0.01)

    sigma            = 1 + A2/A1 + A3/A1
    predicted_points = np.zeros((len(T4) * len(A4), 8))
    pt_num           = 0
    for iT4 in T4:
        for iA4 in A4:
            delta1       = (iA4 / sigma) * 1
            delta2       = (iA4 / sigma) * A2 / A1
            delta3       = (iA4 / sigma) * A3 / A1
            A1_corrected = A1 - delta1
            A2_corrected = A2 - delta2
            A3_corrected = A3 - delta3

            predicted_points[pt_num, 0] = T1
            predicted_points[pt_num, 1] = T2
            predicted_points[pt_num, 2] = T3
            predicted_points[pt_num, 3] = iT4
            predicted_points[pt_num, 4] = A1_corrected
            predicted_points[pt_num, 5] = A2_corrected
            predicted_points[pt_num, 6] = A3_corrected
            predicted_points[pt_num, 7] = iA4
            pt_num += 1

    return predicted_points

def setup_predictions_quadruple_new(N_pts):
    """
    New function to handle the 4 component tuning. I realised that keeping the ratio of the A1:A2:A3 is not necessarily correct. 
    So, instead of tuning double, then triple, then quadruple, I will just try tuning the quadruple one from the start.

    However, to keep the constraint, I am going to have the algorithm predict / sample from 3 angles, and then transform back into
    amplitude space. This works by having a unit radius 4-dimensional hypersphere. The square of each of the cartesian transformed
    variables gives me A1, A2, A3, A4 etc. whilst maintaining the constraint.
    
    Also, previously I was exhaustively finding every combination of input parameters to define the points to make a prediction at.
    Instead, chatGPT says I can uniformally sample points within each parameter's domain to create a matrix of points that efficiently
    covers the domain.

    Thanks to Will Parker and chatGPT for helping develop this!
    """
    
    # define the domains for the time constants t1, t2, t3, t4 and tR
    T1 = [3.0, 8.0]
    T2 = [5, 30]
    T3 = [50, 150]
    T4 = [200, 500]
    tR = [0.4, 1.1]

    # define the domains for the angles theta1, theta2, theta3 defining the unit radius hypersphere
    theta1 = [0, np.pi]
    theta2 = [0, np.pi]
    theta3 = [0, 2 * np.pi]

    # uniformly sample each of these points within the domain to create N test points
    predicted_points = np.zeros((N_pts, 8))
    T1_vals     = np.random.uniform(low = T1[0], high = T1[1], size = N_pts)
    T2_vals     = np.random.uniform(low = T2[0], high = T2[1], size = N_pts)
    T3_vals     = np.random.uniform(low = T3[0], high = T3[1], size = N_pts)
    T4_vals     = np.random.uniform(low = T4[0], high = T4[1], size = N_pts)
    tR_vals     = np.random.uniform(low = tR[0], high = tR[1], size = N_pts)
    theta1_vals = np.random.uniform(low = theta1[0], high = theta1[1], size = N_pts)
    theta2_vals = np.random.uniform(low = theta2[0], high = theta2[1], size = N_pts)
    theta3_vals = np.random.uniform(low = theta3[0], high = theta3[1], size = N_pts)

    # fill in the uniformly sampled points
    predicted_points[:, 0] = T1_vals
    predicted_points[:, 1] = T2_vals
    predicted_points[:, 2] = T3_vals
    predicted_points[:, 3] = T4_vals
    predicted_points[:, 4] = tR_vals
    predicted_points[:, 5] = theta1_vals
    predicted_points[:, 6] = theta2_vals
    predicted_points[:, 7] = theta3_vals

    return predicted_points

def acquisition_function(mu_posterior, cov_posterior, exploration_coeff, predictions, ITERATION):
    """
    Simplest case acquisition function --> upper confidence bound.

    f(x; lambda) = mu(x) + lambda x cov(x)

    Exploitation vs exploration trade off tuned with parameter lambda. Larger values of lambda lead to higher error
    points having more impact on the f(x) value.

    Since we want to MINIMISE the cost function, I need to prioritise: points with LOW mean function and HIGH variance.
    My cost function will never be non-zero (can my expected values be? Maybe?)
    --> f(x; lambda) = - mu(x) + lambda x cov(x)

    f(x; lambda) proportional to the mean of the predicted points + error on them.
    """

    def calc_mean_error(parameter):
        """
        Function used to find the mean cost function in each dimension bin, and return the 
        error on that mean for the plotting functions.
        """

        # use binned statistic to bin the prediction points in a given dimension and return the mean cost in that bin
        # the bin idx for each point is also returned for use in the error calculation
        mean_cost, bins, idx = stats.binned_statistic_dd(parameter, mu_posterior, bins = 20, statistic = "mean")
        idx -= 1 # to get it to match the mids array and make bin[1] be mids[0]
        
        # plot the mid points of each bin
        mids = bins[0][:-1] + np.diff(bins[0][:-1])[0]/2
        
        # use 'error propagation' formula to find the uncertainity on each average value
        bin_error = []
        for ibin in range(len(mids)):
            # find all the variances and means inside this bin
            var_in_bin   = variance[np.where(idx == ibin)[0]]
            print(var_in_bin)
            means_in_bin = mu_posterior[np.where(idx == ibin)[0]]
            # print(var_in_bin.shape)
            # print(means_in_bin.shape)
            # print(var_in_bin)
            # print(means_in_bin)
            
            # calculate the error on this point with the error propagation formula
            err = (mean_cost[ibin] * np.sqrt(np.sum((var_in_bin / means_in_bin)**2))) / len(mids)
            bin_error.append(err)

        print("Mids: \n", mids)
        print("mean_cost: \n", mean_cost)
        print("Error: \n", bin_error)
        return mids, mean_cost, bin_error
    
    # find the variance as the square of the diagonal of each point in the cov matrix
    variance = np.sqrt(np.diag(cov_posterior))
    print("variance:", variance[0:10])
    print(f"Max variance: {max(variance)}\nCorresponding to point: {predicted_points[np.argmax(variance), :]}")
    f = - mu_posterior + exploration_coeff * variance

    # make a 2 D plot to see what's going on...
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    """
    Marginalise out the other dimensions to see the mean, variance, and chosen
    point to sample at in each dimension, for this iteration.
    """

    print("Variance dimensions: ", np.shape(variance))
    print("Mean dimensions: ", np.shape(mu_posterior))
    # x= np.arange(1, len(variance)+1, 1)
    # fig = plt.figure()
    # fig.set_figwidth(20)
    # plt.errorbar(x, mu_posterior, yerr = 2*variance, elinewidth = 0.5, linestyle = "", marker = "o", markersize = 0.5)
    # # plt.yscale("log")
    # plt.axvline(np.argmax(f), linestyle = "--", color = "red", label = f"Sample Position - f{predictions[np.argmax(f), :]}")
    # plt.plot(x, f, label = "AF", color = "red", linewidth = 1)
    # plt.legend()
    
    # plt.savefig(f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/plots/posterior_{ITERATION}_{CURRENT_MODEL}.pdf")
    # plt.close()
    # fig = plt.figure()
    # fig.set_figwidth(20)
    # idx_plot = np.nonzero(f)[0]
    # print(idx_plot)
    # print(len(idx_plot), len(f))
    # plt.plot(x[idx_plot], f[idx_plot])
    # # plt.yscale("log")
    # plt.savefig(f"/data/snoplus3/hunt-stokes/automated_tuning/recursion_dag/plots/acquistion_{ITERATION}_{CURRENT_MODEL}.pdf")
    # plt.close()
    # for i in f:
    #     print(i)
    # print("fmax: ", max(f), "\nFmin: ", min(f))
    # plt.savefig("AF.pdf")
    # find the max of f

    # create histograms of each parameter independently
    t1     = predicted_points[:, 0]
    t2     = predicted_points[:, 1]
    t3     = predicted_points[:, 2]
    t4     = predicted_points[:, 3]
    tr     = predicted_points[:, 4]
    theta1 = predicted_points[:, 5]
    theta2 = predicted_points[:, 6]
    theta3 = predicted_points[:, 7]

    # sort these arrays low --> high, then apply the sorted idx to the variance, mu and acquisition function
    t1_idx         = np.argsort(t1)
    t1_sorted      = np.sort(t1)
    mu_t1_sorted   = mu_posterior[t1_idx]
    var_t1_sorted  = variance[t1_idx]
    acq_t1_sorted  = f[t1_idx]

    t2_idx         = np.argsort(t2)
    t2_sorted      = np.sort(t2)
    mu_t2_sorted   = mu_posterior[t2_idx]
    var_t2_sorted  = variance[t2_idx]
    acq_t2_sorted  = f[t2_idx]

    t3_idx         = np.argsort(t3)
    t3_sorted      = np.sort(t3)
    mu_t3_sorted   = mu_posterior[t3_idx]
    var_t3_sorted  = variance[t3_idx]
    acq_t3_sorted  = f[t3_idx]

    t4_idx         = np.argsort(t4)
    t4_sorted      = np.sort(t4)
    mu_t4_sorted   = mu_posterior[t4_idx]
    var_t4_sorted  = variance[t4_idx]
    acq_t4_sorted  = f[t4_idx]

    tr_idx         = np.argsort(tr)
    tr_sorted      = np.sort(tr)
    mu_tr_sorted   = mu_posterior[tr_idx]
    var_tr_sorted  = variance[tr_idx]
    acq_tr_sorted  = f[tr_idx]

    theta1_idx         = np.argsort(theta1)
    theta1_sorted      = np.sort(theta1)
    mu_theta1_sorted   = mu_posterior[theta1_idx]
    var_theta1_sorted  = variance[theta1_idx]
    acq_theta1_sorted  = f[theta1_idx]

    theta2_idx         = np.argsort(theta2)
    theta2_sorted      = np.sort(theta2)
    mu_theta2_sorted   = mu_posterior[theta2_idx]
    var_theta2_sorted  = variance[theta2_idx]
    acq_theta2_sorted  = f[theta2_idx]

    theta3_idx         = np.argsort(theta3)
    theta3_sorted      = np.sort(theta3)
    mu_theta3_sorted   = mu_posterior[theta3_idx]
    var_theta3_sorted  = variance[theta3_idx]
    acq_theta3_sorted  = f[theta3_idx]

    print("Max Posterior: ", np.max(mu_posterior))
    
    """
    New method to deal with the marginalisation --> binned statistic!
    Calculate the mean of the surrogate across a binned 1D parameter scan.
    This might make the output plots less spikey...
    """

    mids_t1, mean_cost_t1, bin_error_t1             = calc_mean_error(t1)
    mids_t2, mean_cost_t2, bin_error_t2             = calc_mean_error(t2)
    mids_t3, mean_cost_t3, bin_error_t3             = calc_mean_error(t3)
    mids_t4, mean_cost_t4, bin_error_t4             = calc_mean_error(t4)
    mids_tr, mean_cost_tr, bin_error_tr             = calc_mean_error(tr)
    mids_theta1, mean_cost_theta1, bin_error_theta1 = calc_mean_error(theta1)
    mids_theta2, mean_cost_theta2, bin_error_theta2 = calc_mean_error(theta2)
    mids_theta3, mean_cost_theta3, bin_error_theta3 = calc_mean_error(theta3)
    fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (24, 8))
    axes[0,0].plot(mids_t1, mean_cost_t1, color = "black")
    axes[0,0].fill_between(mids_t1, mean_cost_t1 - bin_error_t1, mean_cost_t1 + bin_error_t1, color = 'blue', alpha = 0.2)
    axes[0,0].set_xlabel(r"$T_1$", fontsize = 20)
    axes[0,0].set_ylabel("Mean Posterior", fontsize = 20)
    axes[0,1].plot(mids_t2, mean_cost_t2, color = "black")
    axes[0,1].fill_between(mids_t2, mean_cost_t2 - bin_error_t2, mean_cost_t2 + bin_error_t2, color = 'blue', alpha = 0.2)
    axes[0,1].set_xlabel(r"$T_2$", fontsize = 20)
    axes[0,1].set_ylabel("Mean Posterior", fontsize = 20)
    axes[0,2].plot(mids_t3, mean_cost_t3, color = "black")
    axes[0,2].fill_between(mids_t3, mean_cost_t3 - bin_error_t3, mean_cost_t3 + bin_error_t3, color = 'blue', alpha = 0.2)
    axes[0,2].set_xlabel(r"$T_3$", fontsize = 20)
    axes[0,2].set_ylabel("Mean Posterior", fontsize = 20)
    axes[0,3].plot(mids_t4, mean_cost_t4, color = "black")
    axes[0,3].fill_between(mids_t4, mean_cost_t4 - bin_error_t4, mean_cost_t4 + bin_error_t4, color = 'blue', alpha = 0.2)
    axes[0,3].set_xlabel(r"$T_4$", fontsize = 20)
    axes[0,3].set_ylabel("Mean Posterior", fontsize = 20)
    axes[1,0].plot(mids_tr, mean_cost_tr, color = "black")
    axes[1,0].fill_between(mids_tr, mean_cost_tr - bin_error_tr, mean_cost_tr + bin_error_tr, color = 'blue', alpha = 0.2)
    axes[1,0].set_xlabel(r"$T_R$", fontsize = 20)
    axes[1,0].set_ylabel("Mean Posterior", fontsize = 20)
    axes[1,1].plot(mids_theta1, mean_cost_theta1, color = "black")
    axes[1,1].fill_between(mids_theta1, mean_cost_theta1 - bin_error_theta1, mean_cost_theta1 + bin_error_theta1, color = 'blue', alpha = 0.2)
    axes[1,1].set_xlabel(r"$\theta_1$", fontsize = 20)
    axes[1,1].set_ylabel("Mean Posterior", fontsize = 20)
    axes[1,2].plot(mids_theta2, mean_cost_theta2, color = "black")
    axes[1,2].fill_between(mids_theta2, mean_cost_theta2 - bin_error_theta2, mean_cost_theta2 + bin_error_theta2, color = 'blue', alpha = 0.2)
    axes[1,2].set_xlabel(r"$\theta_2$", fontsize = 20)
    axes[1,2].set_ylabel("Mean Posterior", fontsize = 20)
    axes[1,3].plot(mids_theta3, mean_cost_theta3, color = "black")
    axes[1,3].fill_between(mids_theta3, mean_cost_theta3 - bin_error_theta3, mean_cost_theta3 + bin_error_theta3, color = 'blue', alpha = 0.2)
    axes[1,3].set_xlabel(r"$\theta_3$", fontsize = 20)
    axes[1,3].set_ylabel("Mean Posterior", fontsize = 20)
    fig.tight_layout()
    plt.savefig(f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/plots/marginalised_mean_{ITERATION}.pdf")
    plt.close()


    """
    This creates a plot over every single parameter value in the space...
    """
    fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (24, 8))
    axes[0,0].plot(t1_sorted, mu_t1_sorted, markersize = 5, color = "black", label = "Surrogate")
    axes[0,0].fill_between(t1_sorted, mu_t1_sorted - var_t1_sorted, mu_t1_sorted + var_t1_sorted, color = 'blue', alpha = 0.2, label = "Variance")
    axes[0,0].axvline(t1_sorted[np.argmax(acq_t1_sorted)], linestyle = "dashed", label = "Selected point: " + r"$T_1 = $" + f"{round(t1_sorted[np.argmax(acq_t1_sorted)], 3)}", color = "red")
    axes[0,0].plot(t1_sorted, acq_t1_sorted, color = "green", label = "Acquisition Function")
    axes[0,0].legend()
    axes[0,0].set_xlabel(r"$T_1$ (ns)", fontsize = 20)
    axes[0,0].set_ylabel("Posterior", fontsize = 20)
    
    # axes[0,1].errorbar(t2, mu_posterior, yerr = variance, linestyle = "", marker = "o", markersize= 5, capsize = 2,color = "black")
    axes[0,1].plot(t2_sorted, mu_t2_sorted, markersize = 5, color = "black", label = "Surrogate")
    axes[0,1].fill_between(t2_sorted, mu_t2_sorted - var_t2_sorted, mu_t2_sorted + var_t2_sorted, color = 'blue', alpha = 0.2, label = "Variance")
    axes[0,1].axvline(t2_sorted[np.argmax(acq_t2_sorted)], linestyle = "dashed", label = "Selected point: " + r"$T_2 = $" + f"{round(t2_sorted[np.argmax(acq_t2_sorted)], 3)}", color = "red")
    axes[0,1].plot(t2_sorted, acq_t2_sorted, color = "green", label = "Acquisition Function")
    axes[0,1].legend()
    axes[0,1].set_xlabel(r"$T_2$ (ns)", fontsize = 20)
    axes[0,1].set_ylabel("Posterior", fontsize = 20)
    
    # axes[0,2].errorbar(t3, mu_posterior, yerr = variance, linestyle = "", marker = "o", markersize= 5, capsize = 2,color = "black")
    axes[0,2].plot(t3_sorted, mu_t3_sorted, markersize = 5, color = "black", label = "Surrogate")
    axes[0,2].fill_between(t3_sorted, mu_t3_sorted - var_t3_sorted, mu_t3_sorted + var_t3_sorted, color = 'blue', alpha = 0.2, label = "Variance")
    axes[0,2].axvline(t3_sorted[np.argmax(acq_t3_sorted)], linestyle = "dashed", label = "Selected point: " + r"$T_3 = $" + f"{round(t3_sorted[np.argmax(acq_t3_sorted)], 3)}", color = "red")
    axes[0,2].plot(t3_sorted, acq_t3_sorted, color = "green", label = "Acquisition Function")
    axes[0,2].legend()
    axes[0,2].set_xlabel(r"$T_3$ (ns)", fontsize = 20)
    axes[0,2].set_ylabel("Posterior", fontsize = 20)
    
    # axes[0,3].errorbar(t4, mu_posterior, yerr = variance, linestyle = "", marker = "o", markersize= 5, capsize = 2,color = "black")
    axes[0,3].plot(t4_sorted, mu_t4_sorted, markersize = 5, color = "black", label = "Surrogate")
    axes[0,3].fill_between(t4_sorted, mu_t4_sorted - var_t4_sorted, mu_t4_sorted + var_t4_sorted, color = 'blue', alpha = 0.2, label = "Variance")
    axes[0,3].axvline(t4_sorted[np.argmax(acq_t4_sorted)], linestyle = "dashed", label = "Selected point: " + r"$T_4 = $" + f"{round(t4_sorted[np.argmax(acq_t4_sorted)], 3)}", color = "red")
    axes[0,3].plot(t4_sorted, acq_t4_sorted, color = "green", label = "Acquisition Function")
    axes[0,3].legend()
    axes[0,3].set_xlabel(r"$T_4$ (ns)", fontsize = 20)
    axes[0,3].set_ylabel("Posterior", fontsize = 20)
    
    # axes[1,0].errorbar(tr, mu_posterior, yerr = variance, linestyle = "", marker = "o", markersize= 5, capsize = 2,color = "black")
    axes[1,0].plot(tr_sorted, mu_tr_sorted, markersize = 5, color = "black", label = "Surrogate")
    axes[1,0].fill_between(tr_sorted, mu_tr_sorted - var_tr_sorted, mu_tr_sorted + var_tr_sorted, color = 'blue', alpha = 0.2, label = "Variance")
    axes[1,0].axvline(tr_sorted[np.argmax(acq_tr_sorted)], linestyle = "dashed", label = "Selected point: " + r"$T_r = $" + f"{round(tr_sorted[np.argmax(acq_tr_sorted)], 3)}", color = "red")
    axes[1,0].plot(tr_sorted, acq_tr_sorted, color = "green", label = "Acquisition Function")
    axes[1,0].legend()
    axes[1,0].set_xlabel(r"$T_R$ (ns)", fontsize = 20)
    axes[1,0].set_ylabel("Posterior", fontsize = 20)
    
    # axes[1,1].errorbar(theta1, mu_posterior, yerr = variance, linestyle = "", marker = "o", markersize= 5, capsize = 2,color = "black")
    axes[1,1].plot(theta1_sorted, mu_theta1_sorted, markersize = 5, color = "black", label = "Surrogate")
    axes[1,1].fill_between(theta1_sorted, mu_theta1_sorted - var_theta1_sorted, mu_theta1_sorted + var_theta1_sorted, color = 'blue', alpha = 0.2, label = "Variance")
    axes[1,1].axvline(theta1_sorted[np.argmax(acq_theta1_sorted)], linestyle = "dashed", label = "Selected point: " + r"$\theta_1 = $" + f"{round(theta1_sorted[np.argmax(acq_theta1_sorted)], 3)}", color = "red")
    axes[1,1].plot(theta1_sorted, acq_theta1_sorted, color = "green", label = "Acquisition Function")
    axes[1,1].legend()
    axes[1,1].set_xlabel(r"$\theta_1$", fontsize = 20)
    axes[1,1].set_ylabel("Posterior", fontsize = 20)
    
    # axes[1,2].errorbar(theta2, mu_posterior, yerr = variance, linestyle = "", marker = "o", markersize= 5, capsize = 2,color = "black")
    axes[1,2].plot(theta2_sorted, mu_theta2_sorted, markersize = 5, color = "black", label = "Surrogate")
    axes[1,2].fill_between(theta2_sorted, mu_theta2_sorted - var_theta2_sorted, mu_theta2_sorted + var_theta2_sorted, color = 'blue', alpha = 0.2, label = "Variance")
    axes[1,2].axvline(theta2_sorted[np.argmax(acq_theta2_sorted)], linestyle = "dashed", label = "Selected point: " + r"$\theta_2 = $" + f"{round(theta2_sorted[np.argmax(acq_theta2_sorted)], 3)}", color = "red")
    axes[1,2].plot(theta2_sorted, acq_theta2_sorted, color = "green", label = "Acquisition Function")
    axes[1,2].legend()
    axes[1,2].set_xlabel(r"$\theta_2$", fontsize = 20)
    axes[1,2].set_ylabel("Posterior", fontsize = 20)
    
    # axes[1,3].errorbar(theta3, mu_posterior, yerr = variance, linestyle = "", marker = "o", markersize= 5, capsize = 2,color = "black")
    axes[1,3].plot(theta3_sorted, mu_theta3_sorted, markersize = 5, color = "black", label = "Surrogate")
    axes[1,3].fill_between(theta3_sorted, mu_theta3_sorted - var_theta3_sorted, mu_theta3_sorted + var_theta3_sorted, color = 'blue', alpha = 0.2, label = "Variance")
    axes[1,3].axvline(theta3_sorted[np.argmax(acq_theta3_sorted)], linestyle = "dashed", label = "Selected point: " + r"$\theta_3 = $" + f"{round(theta3_sorted[np.argmax(acq_theta3_sorted)], 3)}", color = "red")
    axes[1,3].plot(theta3_sorted, acq_theta3_sorted, color = "green", label = "Acquisition Function")
    axes[1,3].legend()
    axes[1,3].set_xlabel(r"$\theta_3$", fontsize = 20)
    axes[1,3].set_ylabel("Posterior", fontsize = 20)
    fig.tight_layout()
    plt.savefig(f"/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/plots/marginalised_{ITERATION}.pdf")

    return np.argmax(f)

def kernel(x1, x2, params, model):
    """
    RBF kernel returns covariance (similarity measure) between two points.
    
    INPUTS: x1, x2 --> vector of parameters [t1, t2, A1, A2, ...] etc. to find similarity measure between
            params --> list of kernel hyperparameters [amplitude, scale]
    
    OUTPUT: cov(x1, x2)
    """
    if model == "doubleExponential":
        # construct the matrix of length scales
        param_scales  = np.array([params[1][0], params[1][1], params[1][4], params[1][5]])
        length_matrix = np.diag(param_scales) 
    if model == "tripleExponential":
        param_scales = np.array([params[1][0], params[1][1], params[1][2], params[1][4], params[1][5], params[1][6]])
        length_matrix = np.diag(param_scales) 
    if model == "quadrupleExponential":
        param_scales = np.array(params[1])
        length_matrix = np.diag(param_scales) 
        print(f"Param Scales: {param_scales}\nLength Matrix: {length_matrix}")
    # I have worked through this in my big notebook and I think it makes sense --> going to trust jed with this one for
    # now thouhg, as I haven't got that book with me and Jed wrote this code before
    # correction I've worked through it in my supplementary notebook now, nice.
    # print(x1[0])
    # print(x1[1])
    # print(((x1[0]-x1[1]) / param_scales)**2)
    # print(np.sum(((x1[0]-x1[1]) / param_scales)**2))
    x1 = x1[:, None, :]
    x2 = x2[None, :, :]
    print("Shape of points inputted into kernel: ", x1.shape)
    length_matrix = param_scales[None, None, :]
    print(length_matrix[0,0,0])
    print("Shape of length scale matrix applied: ", length_matrix.shape)
    # print("Length Matrix applied: ", length_matrix)
    print("Shape of subtraction: ", np.shape(x1-x2))
    dX = (x1 - x2) / (2*length_matrix)
    print("Distance between points matrix shape is: ", dX.shape)
    exponent = np.linalg.norm(dX, axis = 2)
    print("EXPONENT: ", exponent)
    print("Exponent after linalg norm is: ", exponent.shape)
    cov = (params[0]**2) * np.exp(- exponent**2)
    print("Shape of cov: ", cov.shape)
    ## actually lets do my nested loop implementation as a I think I understand it more ...
    # cov = np.zeros((x1.shape[0], x2.shape[0]))
    # for i in range(x1.shape[0]):
    #     for j in range(x2.shape[0]):
    #         cov[i,j] = params[0] * np.exp(- np.linalg.norm(x2[i, :]-x1[j, :]) / params[1])

    return cov

def conditional(x_measured, x_predicted, y_measured, params, model):
    """
    Create the conditional PDF, based on a set of measured and predicted points.
    """

    # covariance matrix
    cov_measured      = kernel(x_measured, x_measured, params, model)
    print(cov_measured)
    print("Calculated covariance of measurements!")
    print("Shape of Covariance: ", cov_measured.shape)
    cov_predicted     = kernel(x_predicted, x_predicted, params, model)
    print("Cov predicted: ", cov_predicted)
    print("Calculated covariance of predictions!")
    print("Shape of Covariance: ", cov_predicted.shape)
    cov_pred_measured = kernel(x_predicted, x_measured, params, model)
    print("Calculated covariance between predictions and measurements!")
    print("Shape of Covariance: ", cov_pred_measured.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(cov_measured)
    plt.savefig("cov.png")
    plt.close()

    # add some 'noise' to the measured points (dunno how much lol)
    cov_measured = cov_measured + 0.05 * np.eye(cov_measured.shape[0])
    print(cov_measured)
    # conditional PDF covariance functions
    inv_cov_measured = np.linalg.inv(cov_measured) # expensive step so only do it once --> N observations, order N^3 operation
    print(inv_cov_measured)
    print("Measured points: ", y_measured)
    print("Inverted measurement covariance matrix!")
    print(cov_pred_measured.shape)
    conditional_cov  = cov_predicted - cov_pred_measured.dot(inv_cov_measured.dot(cov_pred_measured.T))
    conditional_mu   = cov_pred_measured.dot(inv_cov_measured) @ y_measured
    print(conditional_cov.shape)
    print(conditional_mu.shape)
    return conditional_mu, conditional_cov

def create_output_files(fname, model, sample_point):
    if os.path.isdir(f"MC/{ISOTOPE}/{fname}"):
        pass
    else:
        os.makedirs(f"MC/{ISOTOPE}/{fname}")
    if os.path.isdir(f"macros/{ISOTOPE}"):
        pass
    else:
        os.makedirs(f"macros/{ISOTOPE}")
    
    # create the template macro
    if model == "doubleExponential":
        with open("template_macro_double.mac", "r") as f:
            rawTextMacro = string.Template(f.read())
        outTextMacro = rawTextMacro.substitute(T1 = round(sample_point[0], 3), T2 = round(sample_point[1], 3), A1 = round(sample_point[2], 3), A2 = round(sample_point[3], 3))
        with open(f"macros/{ISOTOPE}/{fname}.mac", "w") as f:
            f.write(outTextMacro)

    if model == "tripleExponential":
        with open("template_macro_triple.mac", "r") as f:
            rawTextMacro = string.Template(f.read())
        outTextMacro = rawTextMacro.substitute(T1 = round(sample_point[0], 3), T2 = round(sample_point[1], 3), T3 = round(sample_point[2], 3), A1 = round(sample_point[3], 3), A2 = round(sample_point[4], 3), A3 = round(sample_point[5], 3))
        with open(f"macros/{ISOTOPE}/{fname}.mac", "w") as f:
            f.write(outTextMacro)

    if model == "quadrupleExponential":
        with open("template_macro_quadruple.mac", "r") as f:
            rawTextMacro = string.Template(f.read())
        print(sample_point[-1])

        # NEED TO CONVERT FROM THE SURFACE OF THE HYPERSPHERE INTO THE A1, A2, A3, A4
        A1 = np.cos(sample_point[5])**2
        A2 = (np.sin(sample_point[5])*np.cos(sample_point[6]))**2
        A3 = (np.sin(sample_point[5])*np.sin(sample_point[6])*np.cos(sample_point[7]))**2
        A4 = (np.sin(sample_point[5])*np.sin(sample_point[6])*np.sin(sample_point[7]))**2
        print("Sum of the amplitudes given by this method: ", A1+A2+A3+A4)
        outTextMacro = rawTextMacro.substitute(T1 = round(sample_point[0], 3), T2 = round(sample_point[1], 3), T3 = round(sample_point[2], 3), T4 = round(sample_point[3], 3), TR = round(sample_point[4], 3), A1 = round(A1, 3), A2 = round(A2, 3), A3 = round(A3, 3), A4 = round(A4, 3))
        with open(f"macros/{ISOTOPE}/{fname}.mac", "w") as f:
            f.write(outTextMacro)

## algorithm logic starts here ##

# 1. Load the JSON config file
with open("config.JSON", "r") as config_file:
    config = json.load(config_file)

# 2. Extract the current iteration number, kernel hyperparameters, and any previously measured points
ITERATION     = config["CUR_LOOP"]
CURRENT_MODEL = config["TUNING_MODEL"]
KERNEL_PARAMS = [config["RBF_SETTINGS"]["AMPLITUDE"], config["RBF_SETTINGS"]["FEATURE_SCALE"]]
EXPLORE_COEFF = config["RBF_SETTINGS"]["EXPLORE"]
MEASURED_PTS  = config["MEASURED_POINTS"]
MEASURED_VALS = config["COST"]
ISOTOPE       = config["ISOTOPE_USE"]
GLOBAL_BEST   = config["GLOBAL_BEST"] 

# if CURRENT_MODEL == "doubleExponential":
#     predicted_points = setup_predictions_double()
# elif CURRENT_MODEL == "tripleExponential":
#     predicted_points = setup_predictions_triple(GLOBAL_BEST["T1"], GLOBAL_BEST["T2"], GLOBAL_BEST["A1"], GLOBAL_BEST["A2"])
if CURRENT_MODEL == "quadrupleExponential" and ITERATION == 1:
    # generate the uniformly sampled points upon which to compute predictions
    predicted_points = setup_predictions_quadruple_new(10000)

    # save these predictions for future iterations to readout from as a np array 
    # dimensions are (N_points, N_params)
    np.save("/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/predicted_points.npy", predicted_points)

if CURRENT_MODEL == "quadrupleExponential" and ITERATION > 1:
    # predicted points previously generated, so load them up
    # want to keep the same predicted points for each iteration
    predicted_points = np.load("/data/snoplus3/hunt-stokes/automated_tuning/optimiser_v2/predicted_points.npy")

# check what iteration --> if 1st, sample random point
if ITERATION == 1:
    # randomly select one of the predicted points as the sample location
    sample_idx = np.random.randint(low = 0, high = predicted_points.shape[0])
    print(f"Iteration 0! Randomly sampling: {predicted_points[sample_idx, :]}")

    # save this as a ' measured point ' in the config FILE
    if CURRENT_MODEL == "doubleExponential":
        T1 = MEASURED_PTS["T1"]
        T1.append(predicted_points[sample_idx, 0])
        config["MEASURED_POINTS"]["T1"] = T1
        T2 = MEASURED_PTS["T2"]
        T2.append(predicted_points[sample_idx, 1])
        config["MEASURED_POINTS"]["T2"] = T2
        A1 = MEASURED_PTS["A1"]
        A1.append(predicted_points[sample_idx, 2])
        config["MEASURED_POINTS"]["A1"] = A1
        A2 = MEASURED_PTS["A2"]
        A2.append(predicted_points[sample_idx, 3])
        config["MEASURED_POINTS"]["A2"] = A2
        config["CUR_PARAMS"]            = f"{round(T1[-1], 3)}_{round(T2[-1], 3)}_{round(A1[-1], 3)}_{round(A2[-1], 3)}"
    if CURRENT_MODEL == "tripleExponential":
        T1 = MEASURED_PTS["T1"]
        T1.append(predicted_points[sample_idx, 0])
        config["MEASURED_POINTS"]["T1"] = T1
        T2 = MEASURED_PTS["T2"]
        T2.append(predicted_points[sample_idx, 1])
        config["MEASURED_POINTS"]["T2"] = T2
        T3 = MEASURED_PTS["T3"]
        T3.append(predicted_points[sample_idx, 2])
        config["MEASURED_POINTS"]["T3"] = T3
        A1 = MEASURED_PTS["A1"]
        A1.append(predicted_points[sample_idx, 3])
        config["MEASURED_POINTS"]["A1"] = A1
        A2 = MEASURED_PTS["A2"]
        A2.append(predicted_points[sample_idx, 4])
        config["MEASURED_POINTS"]["A2"] = A2
        A3 = MEASURED_PTS["A3"]
        A3.append(predicted_points[sample_idx, 5])
        config["MEASURED_POINTS"]["A3"] = A3
        config["CUR_PARAMS"]            = f"{round(T1[-1], 3)}_{round(T2[-1], 3)}_{round(T3[-1], 3)}_{round(A1[-1], 3)}_{round(A2[-1], 3)}_{round(A3[-1], 3)}"
    if CURRENT_MODEL == "quadrupleExponential":
        T1 = MEASURED_PTS["T1"]
        T1.append(predicted_points[sample_idx, 0])
        config["MEASURED_POINTS"]["T1"] = T1
        T2 = MEASURED_PTS["T2"]
        T2.append(predicted_points[sample_idx, 1])
        config["MEASURED_POINTS"]["T2"] = T2
        T3 = MEASURED_PTS["T3"]
        T3.append(predicted_points[sample_idx, 2])
        config["MEASURED_POINTS"]["T3"] = T3
        T4 = MEASURED_PTS["T4"]
        T4.append(predicted_points[sample_idx, 3])
        config["MEASURED_POINTS"]["T4"] = T4
        TR = MEASURED_PTS["TR"]
        TR.append(predicted_points[sample_idx, 4])
        config["MEASURED_POINTS"]["TR"] = TR
        THETA1 = MEASURED_PTS["THETA1"]
        THETA1.append(predicted_points[sample_idx, 5])
        config["MEASURED_POINTS"]["THETA1"] = THETA1
        THETA2 = MEASURED_PTS["THETA2"]
        THETA2.append(predicted_points[sample_idx, 6])
        config["MEASURED_POINTS"]["THETA2"] = THETA2
        THETA3 = MEASURED_PTS["THETA3"]
        THETA3.append(predicted_points[sample_idx, 7])
        config["MEASURED_POINTS"]["THETA3"] = THETA3
        
        config["CUR_PARAMS"]            = f"{round(T1[-1], 3)}_{round(T2[-1], 3)}_{round(T3[-1], 3)}_{round(T4[-1], 3)}_{round(TR[-1], 3)}_{round(THETA1[-1], 3)}_{round(THETA2[-1], 3)}_{round(THETA3[-1], 3)}"

    # create the output macro for the first iteration!
    create_output_files(config["CUR_PARAMS"], CURRENT_MODEL, predicted_points[sample_idx, :])
    with open("config.JSON", "w") as config_file:
        json.dump(config, config_file, indent = 4)

else:
    # we have measured points already, so use conditioning to inform sample point (bayesian prior --> posterior)
    
    # 1. format the measured points arrays into a 2D matrix of the same format as predicted points
    if CURRENT_MODEL == "doubleExponential":
        T1 = MEASURED_PTS["T1"]
        T2 = MEASURED_PTS["T2"]
        A1 = MEASURED_PTS["A1"]
        A2 = MEASURED_PTS["A2"]

        # num measured points x num_features
        measured_points = np.zeros((len(T1), 4))
        measured_points[:, 0] = T1
        measured_points[:, 1] = T2
        measured_points[:, 2] = A1
        measured_points[:, 3] = A2
    if CURRENT_MODEL == "tripleExponential":
        T1 = MEASURED_PTS["T1"]
        T2 = MEASURED_PTS["T2"]
        T3 = MEASURED_PTS["T3"]
        A1 = MEASURED_PTS["A1"]
        A2 = MEASURED_PTS["A2"]
        A3 = MEASURED_PTS["A3"]

        # num measured points x num_features
        measured_points = np.zeros((len(T1), 6))
        measured_points[:, 0] = T1
        measured_points[:, 1] = T2
        measured_points[:, 2] = T3
        measured_points[:, 3] = A1
        measured_points[:, 4] = A2
        measured_points[:, 5] = A3
    if CURRENT_MODEL == "quadrupleExponential":
        T1 = MEASURED_PTS["T1"]
        T2 = MEASURED_PTS["T2"]
        T3 = MEASURED_PTS["T3"]
        T4 = MEASURED_PTS["T4"]
        TR = MEASURED_PTS["TR"]
        THETA1 = MEASURED_PTS["THETA1"]
        THETA2 = MEASURED_PTS["THETA2"]
        THETA3 = MEASURED_PTS["THETA3"]

        # num measured points x num_features
        measured_points = np.zeros((len(T1), 8))
        measured_points[:, 0] = T1
        measured_points[:, 1] = T2
        measured_points[:, 2] = T3
        measured_points[:, 3] = T4
        measured_points[:, 4] = TR
        measured_points[:, 5] = THETA1
        measured_points[:, 6] = THETA2
        measured_points[:, 7] = THETA3

    print("Dimensions of measured points: ", measured_points.shape)
    print("Dimensions of predicted points: ", predicted_points.shape)
    
    # 2. create the conditional PDF, returning the mean and covariance at each predicted point, given measured values
    mu_posterior, cov_posterior = conditional(measured_points, predicted_points, MEASURED_VALS, KERNEL_PARAMS, CURRENT_MODEL)
    print("Calculated the posterior!")
    print(f"Length of mean Posterior: {mu_posterior.shape}\nCovariance: {cov_posterior.shape}")
    # 3. pass this to the ACQUISITION FUNCTION to decide the best point to sample from next, given this conditional PDF
    predicted_point_to_sample_idx = acquisition_function(mu_posterior, cov_posterior, EXPLORE_COEFF, predicted_points, ITERATION)
    next_sample_point             = predicted_points[predicted_point_to_sample_idx, :] 
    print("Found next sample point!")

    # 4. add the next sample point to the ' measured points ' array and create the necessary macros, sh and submit files
    if CURRENT_MODEL == "doubleExponential":
        print(next_sample_point)
        T1.append(next_sample_point[0])
        T2.append(next_sample_point[1])
        A1.append(next_sample_point[2])
        A2.append(next_sample_point[3])
        config["MEASURED_POINTS"]["T1"] = T1
        config["MEASURED_POINTS"]["T2"] = T2
        config["MEASURED_POINTS"]["A1"] = A1
        config["MEASURED_POINTS"]["A2"] = A2
        config["CUR_PARAMS"]            = f"{round(T1[-1], 3)}_{round(T2[-1], 3)}_{round(A1[-1], 3)}_{round(A2[-1], 3)}" 
    if CURRENT_MODEL == "tripleExponential":
        T1.append(next_sample_point[0])
        T2.append(next_sample_point[1])
        T3.append(next_sample_point[2])
        A1.append(next_sample_point[3])
        A2.append(next_sample_point[4])
        A3.append(next_sample_point[5])
        config["MEASURED_POINTS"]["T1"] = T1
        config["MEASURED_POINTS"]["T2"] = T2
        config["MEASURED_POINTS"]["T3"] = T3
        config["MEASURED_POINTS"]["A1"] = A1
        config["MEASURED_POINTS"]["A2"] = A2
        config["MEASURED_POINTS"]["A3"] = A3
        config["CUR_PARAMS"]            = f"{round(T1[-1], 3)}_{round(T2[-1], 3)}_{round(T3[-1], 3)}_{round(A1[-1], 3)}_{round(A2[-1], 3)}_{round(A3[-1], 3)}"
    if CURRENT_MODEL == "quadrupleExponential":
        T1.append(next_sample_point[0])
        T2.append(next_sample_point[1])
        T3.append(next_sample_point[2])
        T4.append(next_sample_point[3])
        TR.append(next_sample_point[4])
        THETA1.append(next_sample_point[5])
        THETA2.append(next_sample_point[6])
        THETA3.append(next_sample_point[7])
        config["MEASURED_POINTS"]["T1"] = T1
        config["MEASURED_POINTS"]["T2"] = T2
        config["MEASURED_POINTS"]["T3"] = T3
        config["MEASURED_POINTS"]["T4"] = T4
        config["MEASURED_POINTS"]["TR"] = TR
        config["MEASURED_POINTS"]["THETA1"] = THETA1
        config["MEASURED_POINTS"]["THETA2"] = THETA2
        config["MEASURED_POINTS"]["THETA3"] = THETA3
        print(f"The chosen point was: {round(T1[-1], 3)}_{round(T2[-1], 3)}_{round(T3[-1], 3)}_{round(T4[-1], 3)}_{round(TR[-1], 3)}_{round(THETA1[-1], 3)}_{round(THETA2[-1], 3)}_{round(THETA3[-1], 3)}")
        config["CUR_PARAMS"]            = f"{round(T1[-1], 3)}_{round(T2[-1], 3)}_{round(T3[-1], 3)}_{round(T4[-1], 3)}_{round(TR[-1], 3)}_{round(THETA1[-1], 3)}_{round(THETA2[-1], 3)}_{round(THETA3[-1], 3)}"

    # create the macros, sh and submit files of relevance
    create_output_files(config["CUR_PARAMS"], CURRENT_MODEL, next_sample_point)
    with open("config.JSON", "w") as config_file:
        json.dump(config, config_file, indent = 4)
    print("Updated Measured values and created macros!")