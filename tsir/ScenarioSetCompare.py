""" Extrapolate.py

Fit a basic TSIR model and extrapolate under different scenarios (i.e. campaign timings and
RI increases and decreases, etc.). """
import sys

## Standard imports
import numpy as np
import pandas as pd

## TSIR model functions
import utils.tsir as tsir

## For R2 scores
from sklearn.metrics import r2_score

def compute_adj_births(df):
    return df["births"]*(1.-0.9*df["mcv1"]*(1.-df["mcv2"])-0.99*df["mcv1"]*df["mcv2"])  

def ExtrapolateBasicTSIR(df,model,num_samples=10000):

    """ Use the basic TSIR model, constructed in BasicTSIR.py, to extrapolate. df is assumed to be the
    tsir_df output by model fitting concatenated with adj_births, sia, etc. necessary for extrapolation. 
    Amount of extrapolation time is dictated by df.index, and model is the dictionary constructed by model 
    fitting. """

    ## Hyper parameters
    n_steps = len(df)
    n_steps_with_data = len(df["cases"].dropna())
    std_logE = model["std_logE"]

    ## Allocate the appropriate storage
    I_samples = np.zeros((num_samples,n_steps))
    S_samples = np.zeros((num_samples,n_steps))

    ## Set up the initial conditions, accounting for importation
    I_samples[:,0] = df["I_t"].values[0]
    S_samples[:,0] = model["S_bar"] + df["Z_t"].values[0]

    ## Loop through time
    for i in range(1,n_steps):
        
        ## Time of year for seasonality
        time_in_period = i % model["periodicity"]

        ## If we have data, compute the one-step projection. Otherwise, 
        ## extrapolate freely. On the final timestep with data, we have no way to
        ## infer importation pressure, so we assume it is zero.
        if i < n_steps_with_data:
            lam = model["t_beta"][time_in_period]*(S_samples[:,i-1])*(df["I_t"].values[i-1]**model["alpha"])
        elif i == n_steps_with_data:
            lam = model["t_beta"][time_in_period]*(S_samples[:,i-1])*(df["I_t"].values[i-1]**model["alpha"])
        else:
            lam = model["t_beta"][time_in_period]*(S_samples[:,i-1])*(I_samples[:,i-1]**model["alpha"])

        ## Update accordingly
        I_samples[:,i] = lam*np.exp(std_logE*np.random.normal(size=(num_samples,)))
        S_samples[:,i] = (S_samples[:,i-1]+df["adj_births"].values[i]-I_samples[:,i])*(1.-df["sia"].values[i-1])

        ## Take care of negatives (This happens for large std_logE, which probably shouldn't
        ## be the case? I need a better fix...)
        I_samples[I_samples[:,i] < 0,i] = 0.

    return I_samples, S_samples

def ComputeBurdenEstimate(I_samples,time,start_time,end_time):

    """ Wrapper function for simple total burden estimates from I_samples. """

    start_idx = np.argmin([np.abs(d-start_time) for d in time])
    end_idx = np.argmin([np.abs(d-end_time) for d in time])
    total_samples = I_samples[:,start_idx:end_idx].sum(axis=1)
    avg_total = np.mean(total_samples)
    std_total = np.std(total_samples)
    low_total = np.percentile(total_samples,5.)
    high_total = np.percentile(total_samples,95.)

    return avg_total, std_total, low_total, high_total

if __name__ == "__main__":

    ## Get the dataset
    country = "chad"
    df = pd.read_pickle("..\\outputs\\modeling_dataset.pkl")
    df = df.loc[country]
    df = df.loc["2014-01-01":]

    ## Print the dataset
    print("\nInput dataset for {}".format(country.title()))
    print(df)

    ## Construct the province level TSIR model.
    print("\nFitting the model...")
    tsir_df, model, rr = tsir.FitTSIRModel(df.copy(),
                                           tsir.BasicSusceptibleReconstruction,
                                           tsir.BasicTransmissionRegression)
    print("...done!")

    ## Print some model diagnostics
    print("\nSome model parameters:")
    print("Reporting rate = {}".format(rr))
    print("Alpha estimate is {} +/- {}".format(model["alpha"],model["alpha_std"]))
    print("S_bar = {} +/- {}".format(model["S_bar"],model["S_bar_std"]))
    print("Beta estimate:")
    print(model["scale_factor"]*model["t_beta"])
    print("sigma_epsilon = {}".format(model["std_logE"]))
    print("SIA efficacies = ")
    print(tsir_df.loc[tsir_df["target_pop"] != 0.,["target_pop","sia"]])

    ## Construct the extrapolation DF by first reindexing the dataframe
    ## to the extrapolation time we want, then filling NaNs in the adj_births and
    ## SIA columns to pick a particular extrapolation scenario.
    time = pd.date_range(start=tsir_df.index[0],end="2045-12-31",freq="SM")
    tsir_df = tsir_df.reindex(time)
    
    ## Fill NaNs to create the appropriate extrapolation scenario.
    sia_test_times = pd.date_range(start="2035-01-01",end="2036-12-31",freq="SM")[::2]
    print("SIA test times:")
    print(sia_test_times)
    burden_window = 5
    print("SIA burden window = {} years".format(burden_window))

    ## Make a baseline
    baseline = tsir_df.copy()
    baseline["adj_births"] = baseline["adj_births"].fillna(method="ffill")
    baseline["sia"] = baseline["sia"].fillna(0.)

    ## And sample the baseline (set seed for reproducibility reasons)
    np.random.seed(23)  
    baseline_I, baseline_S = ExtrapolateBasicTSIR(baseline,model)
    
    ## Loop over SIAs and collect some stats
    scenario_comps = []
    for i, date in enumerate(sia_test_times):

        ## Make the scnario
        this_df = tsir_df.copy()
        this_df["adj_births"] = this_df["adj_births"].fillna(method="ffill")
        this_df["target_pop"] = this_df["target_pop"].fillna(0.)
        this_df["sia"] = this_df["sia"].fillna(0.)
        mu = this_df.loc[this_df["sia"] != 0, "sia"].mean()
        this_df.loc[date,"sia"] = mu

        ## Summarize the extrapolation scenario
        print("\nExtrapolation scenario = {}".format(i))
        print(this_df[["adj_births","mcv1","mcv2","births"]])
        print(this_df.loc[this_df["sia"] != 0,"sia"])

        ## Use the extrapolation method above to compute model forecasts
        np.random.seed(23)
        I_samples, S_samples = ExtrapolateBasicTSIR(this_df,model)

        ## Set the comparison times
        start_time = date
        if start_time.day == 29:
            end_time = pd.to_datetime("{}-{}".format(start_time.year+burden_window,
                                                     start_time.strftime("%m-28")))
        else:
            end_time = pd.to_datetime("{}-{}".format(start_time.year+burden_window,
                                                     start_time.strftime("%m-%d")))

        ## Compute the susceptibility at that time
        sia_idx = np.argmin([np.abs(d-start_time) for d in time])
        avg_S = np.mean(baseline_S[:,sia_idx:sia_idx+2])
        std_S = np.std(baseline_S[:,sia_idx:sia_idx+2])
        low_S = np.percentile(baseline_S[:,sia_idx:sia_idx+2],5.)
        high_S = np.percentile(baseline_S[:,sia_idx:sia_idx+2],95.)
        
        ## Compute some estimates
        avg_base, std_base, _, _ = ComputeBurdenEstimate(baseline_I,time,start_time,end_time)
        avg_total, std_total, _, _ = ComputeBurdenEstimate(I_samples,time,start_time,end_time)
        averted = (baseline_I-I_samples)
        avg_averted, std_averted, low_av, high_av  = ComputeBurdenEstimate(averted,time,start_time,end_time)

        ## Store it
        scenario_comps.append((date,
                               avg_S,std_S,low_S,high_S,
                               avg_base,std_base,
                               avg_total,std_total,
                               avg_averted,std_averted,low_av,high_av,
                               ))


    ## Reshape
    scenario_comps = pd.DataFrame(scenario_comps,
                                  columns=["sia_date",
                                            "avg_S","std_S","low_S","high_S",
                                            "avg_base","std_base",
                                            "avg_total","std_total",
                                            "avg_averted","std_averted","low_av","high_av",
                                            ])
    scenario_comps.to_csv("..\\data\\{}_sia_comparisons.csv".format(country.replace(" ","")))
    print("\nFinal output:")
    print(scenario_comps)
    
