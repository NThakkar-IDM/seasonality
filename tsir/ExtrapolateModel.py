""" Extrapolate.py

Fit a basic TSIR model and extrapolate under different scenarios (i.e. campaign timings and
RI increases and decreases, etc.). """
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## TSIR models, visualization 
## utitlies
import utils.tsir as tsir
from utils.vis import *

## For R2 scores
from sklearn.metrics import r2_score
np.random.seed(23)

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
    low_total = np.percentile(total_samples,2.5)
    high_total = np.percentile(total_samples,97.5)

    return avg_total, std_total, low_total, high_total

def axes_setup(axes):
    axes.spines["left"].set_position(("axes", -0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return axes

if __name__ == "__main__":

    ## Get the dataset
    country = "nepal"
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
    ## 0: No SIA, RI as is on latest data point
    ## 1: SIA at a specified time (based on WHO cal), RI as is
    _scenario = 0
    if _scenario == 0:
        tsir_df["adj_births"] = tsir_df["adj_births"].fillna(method="ffill")
        tsir_df["sia"] = tsir_df["sia"].fillna(0.)
        vimc_fname = None
    elif _scenario == 1:
        tsir_df["adj_births"] = tsir_df["adj_births"].fillna(method="ffill")
        tsir_df["sia"] = tsir_df["sia"].fillna(0.)
        tsir_df.loc["2030-11-15","sia"] = tsir_df["sia"].max()
        vimc_fname = None

    ## Summarize the extrapolation scenario
    print("\nExtrapolation scenario = {}".format(_scenario))
    print(tsir_df[["adj_births","mcv1","mcv2","births"]])
    print(tsir_df.loc[tsir_df["sia"] != 0,"sia"])

    ## Use the extrapolation method above to compute model forecasts
    print("\nComputing forecasts...")
    I_samples, S_samples = ExtrapolateBasicTSIR(tsir_df,model)

    ## Compute summary statistics
    low_I, mid_I, high_I = low_mid_high(I_samples)
    low_S, mid_S, high_S = low_mid_high(S_samples)

    ## Trajectories plot
    fig, axes = plt.subplots(2,1,sharex=True,figsize=(12,9))
    for ax in axes:
        ax = axes_setup(ax)
    trajs = [3,538,2282,9858]
    cmap = plt.get_cmap("plasma")
    traj_c = [cmap(x) for x in np.linspace(0.,0.8,len(trajs))]
    for i, t in enumerate(trajs):
        axes[0].plot(time,I_samples[t],lw=2,color=traj_c[i])
        axes[1].plot(time,S_samples[t],lw=2,color=traj_c[i])
    axes[0].plot(time,mid_I,color="k",lw=3)
    axes[0].plot(time[:len(tsir_df["cases"].dropna())],
                 tsir_df["I_t"].values[:len(tsir_df["cases"].dropna())],
                 color="k",marker=".",ls="None",
                 markeredgecolor="k",markerfacecolor="None",
                 #markersize=12,lw=1,
                 )
    axes[1].plot(time,mid_S,color="k",lw=3)
    axes[0].set_ylabel("Infectious people")
    axes[1].set_ylabel("Susceptible people")
    axes[0].ticklabel_format(style='sci',scilimits=(0,0),axis="y")
    axes[1].ticklabel_format(style='sci',scilimits=(0,0),axis="y")
    fig.tight_layout()
    fig.savefig("..\\outputs\\volatility_illustration.png")

    ## Plot the results
    fig, axes = plt.subplots(2,1,sharex=True,figsize=(12,9))

    ## Plot SIA times
    for x in tsir_df.loc[(~tsir_df["target_pop"].isin({0,np.nan})) | (tsir_df["sia"]!=0.)].index:
        axes[0].axvline(x,c="k",alpha=0.4,ls="dashed")
        if x >= df.index[-1]:
            axes[1].axvline(x,c="k",alpha=0.7,ls="dashed")
    axes[1].plot([],alpha=0.7,ls="dashed",color="k",label="{} SIA".format(x.strftime("%B %Y")))
    axes[0].plot([],c="k",alpha=0.4,ls="dashed",label="SIA")

    ## Plot the model fit
    n_steps = len(tsir_df["cases"].dropna())
    axes[0].fill_between(time[:n_steps],low_S[:n_steps],high_S[:n_steps],color="C4",alpha=0.3)
    axes[0].plot(time[:n_steps],mid_S[:n_steps],color="C4",label=r"S$_{t}\,|\,$S$_{t-1}$")
    axes[1].fill_between(time[:n_steps],low_I[:n_steps],high_I[:n_steps],color="#FF420E",alpha=0.3)
    axes[1].plot(time[:n_steps],mid_I[:n_steps],color="#FF420E",label=r"I$_{t}\,|\,$I$_{t-1}$")

    ## Plot the extrapolation
    axes[0].fill_between(time[n_steps:],low_S[n_steps:],high_S[n_steps:],color="#68829E",alpha=0.3)
    axes[0].plot(time[n_steps:],mid_S[n_steps:],color="#68829E",label=r"S$_{t}$ forecast")
    axes[1].fill_between(time[n_steps:],low_I[n_steps:],high_I[n_steps:],color="#68829E",alpha=0.3)
    axes[1].plot(time[n_steps:],mid_I[n_steps:],color="#68829E",label=r"I$_{t}$ forecast")

    ## Plot the data
    axes[1].plot(time[:n_steps],tsir_df["I_t"].values[:n_steps],color="k",marker=".",ls="None")#,label="Scaled case reports")

    ## Make the legends
    axes[0].legend(loc=2)
    axes[1].legend(loc=2)

    ## Axis details
    axes[1].set(ylabel="Infecteds")
    axes[0].set(ylabel="Susceptibles")
    axes[0].ticklabel_format(axis="y",style="sci",scilimits=(0,1))
    axes[1].ticklabel_format(axis="y",style="sci",scilimits=(0,1))
    axes[1].set_xlim((pd.to_datetime("2019-01-01"),time[-1]))
    fig.tight_layout()
    
    ## Compute some R2 values
    print("Short extrapolation R2 = {}".format(r2_score(tsir_df["I_t"].values[:n_steps],mid_I[:n_steps])))

    ## Add some columns to the TSIR DF and save it for use elsewhere
    extrap_df = pd.DataFrame(np.array([low_I, mid_I, high_I,
                                       low_S, mid_S, high_S]).T,
                             index=tsir_df.index,
                             columns=["low_I","mid_I","high_I",
                                      "low_S","mid_S","high_S"])
    tsir_df = pd.concat([tsir_df,extrap_df],axis=1)
    print("\nSaved output:")
    print(tsir_df)
    tsir_df.reset_index().to_csv(
       "..\\data\\{}_tsir_df.csv".format(country.replace(" ",""))
       )

    ## Finish up
    plt.show()
