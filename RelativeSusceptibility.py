""" RelativeSusceptibility.py

Comparing stability based susceptibility estimates to bespoke TSIR models."""

import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Data processing
from Profiles import ProfileRegression
from utils import process_case_data,\
                  process_sia_calendar,\
                  axes_setup

## Plot environment
plt.rcParams["font.size"] = 22.0
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.serif"] = ["Garamond", "Times New Roman"]
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.labelsize"] = "medium"
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"

def demographic_pressure(profiles,stability_extent=1):

    ## Compute the It greens function
    A0 = np.tril(np.ones((12,12)),k=-1) 
    gf = np.exp(A0 @ (2.*profiles.mu_hat + profiles.sigs**2))
    gf = np.vstack(stability_extent*[gf])

    ## Accumulate and estimate
    A1 = np.tril(np.ones((len(gf),len(gf))),k=0)
    A1[:,0] = 0 
    X = np.array([np.ones((len(gf),)),
                  np.arange(len(gf))]).T
    alphas = np.linalg.inv(X.T @ X) @ X.T @ A1 @ gf

    ## Reshape
    alphas = pd.DataFrame(alphas,
        columns=profiles.ln_cr.columns,
        ).T

    return alphas

def relative_susceptibility(alphas,
                            cases,
                            vax,
                            residual):

    ## Get the number of SIAs in the window
    num_sias = len(vax.loc[vax != 0])

    ## Set up the design matrix
    X = np.array([np.ones((len(cases),)),
                  np.arange(len(cases))]).T
    A1 = np.tril(np.ones((len(cases),len(cases))),k=0)
    A1[:,0] = 0 
    demo_presure = (X @ alphas)[:-1]
    cum_cases = A1[:-1,:-1] @ cases.values[1:]
    cum_vax = A1[:-1,:-1] @ vax.values[:-1]
    if num_sias != 0:
        Xr = np.hstack([cum_cases[:,None],cum_vax[:,None]])
    else:
        Xr = cum_cases[:,None]

    ## Solve the problem
    thetaL = np.linalg.inv(Xr.T @ Xr)
    thetas = thetaL @ Xr.T @ demo_presure
    theta_cov = (np.sum((demo_presure - (Xr @ thetas))**2)\
            /(len(demo_presure)))*thetaL

    ## Compute the fluctuations
    Zt = demo_presure - (Xr @ thetas)
    Ztcov = Xr @ theta_cov @ Xr.T
    Zterr = np.sqrt(np.diag(Ztcov))

    ## Use the seasonality model residuals 
    ## to rescale the susceptibility estimate
    vol_scale = np.sum(Zt*(np.exp(residual)-1.))/np.sum(Zt**2)
    Zt *= vol_scale
    Zterr *= vol_scale

    ## Check stability
    success = (thetas > 0).all() & (vol_scale > 0)

    return success, thetas, vol_scale, Zt, Zterr

if __name__ == "__main__":

    ## countries (plotting built for only 3)
    countries_list = ["Chad","Kenya","Nepal"]
    start_date = "2014-01-01"

    ## get the data
    tsir_dfs = {}
    for c in countries_list:
        tsir_df = pd.read_csv("data\\{}_tsir_df.csv".format(c.lower()),
                index_col=0,
                parse_dates=["index"],
                date_parser=pd.to_datetime,
                ).set_index("index")
        tsir_dfs[c] = tsir_df

    # Process raw case data
    data = process_case_data(
        "data\\measlescasesbycountrybymonth_Mar2024.csv",
        long_return=True,
        countries_list=countries_list,
    )
    end_date = "2024-01-01"
    data = data.loc[data["time"] <= end_date]

    ## Get the SIA dose data
    sia_cal = process_sia_calendar("data\\Summary_MR_SIA.csv")

    ## Fit the seasonality model
    profiles = ProfileRegression(data)
    i_to_c = {i:c for i,c in enumerate(profiles.ln_cr.columns)}
    c_to_i = {c:i for i,c in i_to_c.items()}
    residuals = (profiles.ln_cr - profiles.X @ profiles.mu_hat)

    ## Compute demographic pressure parameters via
    ## endemic stability
    pressure = demographic_pressure(profiles,1)
    print("\nDemographic pressure parameters:")
    print(pressure)

    ## Prepare for a loop over countries to reconstruct 
    ## susceptibility
    ## Use vaccines and cases to compute sink terms
    cases = data[["Country","time","Cases"]]\
            .copy()\
            .pivot(columns="Country",
                index="time",
                values="Cases")
    cases = cases.loc[start_date:][profiles.ln_cr.columns]

    ## Make the figure layout
    fig, axes = plt.subplots(2,2,figsize=(14,7.5),sharex=True)
    axes = axes.reshape(-1)
    for ax in axes:
        axes_setup(ax)

    ## Loop over countries, compute and plot
    for i, country in enumerate(countries_list):

        ## Compute the WHO rec. First get the most recent SIA
        cal = sia_cal.loc[(sia_cal["Country"] == country) &\
                          (sia_cal["time"].notnull())]

        ## Create a SIA vax covariate
        vax = cal[["time","doses"]]\
            .dropna()\
            .sort_values("time")\
            .set_index("time")["doses"]
        vax = pd.DataFrame(np.diag(vax.values),
                        index=vax.index,columns=vax.index)
        vax = vax.resample("MS").sum()
        vax.index = vax.index + pd.to_timedelta(14,unit="d")
        vax = vax.reindex(cases.index)\
                .fillna(0).sum(axis=1)/1.e6

        ## Compute the relative susceptibility estimate, recomputing across years
        ## until you find a stable solution.
        success, _, vol_scale, Zt, Zterr = relative_susceptibility(
            pressure.loc[country].values,
            cases[country],
            vax,
            residuals.loc[cases.index[0]:cases.index[-2],country]
            )

        ## Prepare full TSIR outputs for comparison
        tsir_df = tsir_dfs[country]
        St = tsir_df.loc[cases.index[0]:cases.index[-1],
                    ["low_S","mid_S","high_S"]]
        avg_S = St["mid_S"].mean()
        St = (St-avg_S)/avg_S
        
        ## Plot the result
        axes[i].fill_between(cases.index[:-1],Zt-2.*Zterr,Zt+2.*Zterr,
                             facecolor="#0955FE",edgecolor="None",alpha=0.4,zorder=4)
        axes[i].plot(cases.index[:-1],Zt,color="#0955FE",lw=5,label="Reconstruction",zorder=5)
        
        ## And the full model
        axes[i].fill_between(St.index,
                          St["low_S"].values,St["high_S"].values,
                          facecolor="k",edgecolor="None",alpha=0.2,zorder=1)
        axes[i].plot(St["mid_S"],color="grey",lw=2,label="Full model",zorder=2)
        
        ## Make negative space for timeline stuff
        ylim = axes[i].get_ylim()
        axes[i].set_ylim((1.5*ylim[0],1.5*ylim[1]))
        ylim = axes[i].get_ylim()

        ## Add cases for context
        scaled_cases = cases[country]/(cases[country].max())
        axes[i].fill_between(cases.index,ylim[0],
                          (0.5*ylim[1]*(scaled_cases.values)+ylim[0]),
                          facecolor="k",edgecolor="None",alpha=0.85,zorder=1)

        ## Add the campaigns
        sias = vax.loc[vax != 0].index
        for d in sias:
            axes[i].axvline(d,ymin=0,ymax=0.125,lw=3,color="#FE09D0")
        axes[i].plot([],lw=3,color="#FE09D0",label="Campaign",zorder=3)

        ## Tighten
        axes[i].set_ylim(ylim)
        if i%2 == 0:
            axes[i].set_ylabel("Relative susceptibility")
        axes[i].text(
            0.025,
            0.9,
            country.title(),
            fontsize=26,
            color="k",
            transform=axes[i].transAxes,
            )
    
    ## Make a legend?
    axes[-1].axis("off")
    axes[-1].plot([],color="#0955FE",lw=5,label="Estimate based on the\nseasonality profile")
    axes[-1].fill_between([],[],[],facecolor="grey",alpha=0.3,edgecolor="None",
                          label="Estimate from the full\ntransmission model")
    axes[-1].fill_between([],[],[],
                    facecolor="k",edgecolor="None",alpha=0.85,
                    label="Timeline of case reports",
                    )
    axes[-1].plot([],lw=3,color="#FE09D0",label="Vaccination campaign times",zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    order = [0,2,3,1]
    axes[-1].legend([handles[i] for i in order],
                    [labels[i] for i in order],
                    loc="center",frameon=False)

    ## Finish up
    xticks = pd.date_range("2015-01-01","2023-01-01",freq="2YS")
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels([x.year for x in xticks])

    ## Finish up
    fig.tight_layout()
    fig.savefig("outputs\\multi_susceptible_recon.png")
    plt.show()