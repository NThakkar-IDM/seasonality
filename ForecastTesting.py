""" ForecastTesting.py

Post-campaign linear extrapolation tests for a few countries. """

import os

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Data processing
from Profiles import ProfileRegression
from utils import process_case_data,\
                  process_sia_calendar,\
                  axes_setup
from collections import defaultdict

## Susceptibility inference
from RelativeSusceptibility import demographic_pressure,\
                                   relative_susceptibility

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

def get_outbreak_months(low_season_p, 
                        model_type="LogNormal",
                        key_conv = {},
                        ):
    pr_Re_geq_1 = 1 - low_season_p
    indices = np.argwhere(pr_Re_geq_1 >= 0.8)
    out = defaultdict(list)
    for v, k in indices:
        out[key_conv.get(k,k)].append(v+1)
    return out

if __name__ == "__main__":

    _example = 1
    if _example == 1:
        countries_list = ["Pakistan"]
        start_date = "2014-01-01"
        end_date = "2024-01-01"
        training_date = "2018-12-01"
        extrap_months = 36
    if _example == 2:
        countries_list = ["Kenya"]
        start_date = "2014-01-01"
        end_date = "2024-01-01"
        training_date = "2021-08-01"
        extrap_months = 36
    if _example == 3:
        countries_list = ["Chad"]
        start_date = "2014-01-01"
        end_date = "2024-01-01"
        training_date = "2021-05-01"
        extrap_months = 24
    if _example == 4:
        countries_list = ["Nepal"]
        start_date = "2014-01-01"
        end_date = "2024-01-01"
        training_date = "2020-04-01"
        extrap_months = 36

    # Process raw case data
    data = process_case_data(
        os.path.join("data","measlescasesbycountrybymonth_Mar2024.csv"),
        long_return=True,
        countries_list=countries_list,
    )
    data = data.loc[data["time"] <= end_date]
    tdata = data.loc[data["time"] <= training_date].copy()

    ## Get the SIA dose data
    sia_cal = process_sia_calendar(os.path.join("data","Summary_MR_SIA.csv"))

    ## Fit the seasonality model
    logt = ProfileRegression(tdata)
    i_to_c = {i:c for i,c in enumerate(logt.ln_cr.columns)}
    residuals = (logt.ln_cr - logt.X @ logt.mu_hat)

    ## Compute high-season months
    outbreak_months = get_outbreak_months(logt.p_low,
                        key_conv=i_to_c)[countries_list[0]]

    ## Use vaccines and cases to compute sink terms
    tcases = tdata[["time","Cases"]]\
            .copy()\
            .sort_values("time")\
            .set_index("time")["Cases"]
    tcases = tcases.loc[start_date:]
    cases = data[["time","Cases"]]\
            .copy()\
            .sort_values("time")\
            .set_index("time")["Cases"]
    cases = cases.loc[start_date:]

    ## And vax
    cal = sia_cal.loc[sia_cal["Country"] == countries_list[0]]
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
    tvax = vax.loc[:tcases.index[-1]].copy()

    ## Compute demographic pressure parameters via
    ## endemic stability
    pressure = demographic_pressure(logt,1)
    alphas = pressure.values[0,:]

    ## Then estimate relative susceptibility
    success, thetas, vol_scale, Zt, Zterr = relative_susceptibility(
            alphas,
            tcases,
            tvax,
            residuals.loc[tcases.index[0]:tcases.index[-2],countries_list[0]]
            )

    ## Make the forecast
    Zt = pd.Series(Zt,index=tcases.index[:-1])
    Zterr = pd.Series(Zterr,index=tcases.index[:-1])
    ex_time = cases.loc[tcases.index[-1]:].index[:extrap_months]
    Zt_extrap = pd.Series(Zt.values[-1]+np.arange(len(ex_time))*alphas[1]*vol_scale,
                          index=ex_time).iloc[1:]
    Zt_extrap += -vol_scale*np.cumsum(vax.loc[ex_time])*thetas[1]
    Zterr_extrap = 0*Zt_extrap + Zterr.values[-1]

    ## Get the high seasons
    high_seasons = pd.Series(Zt_extrap.index.month.isin(outbreak_months),
                      index=Zt_extrap.index).astype(int)
    
    ## Plot the results
    fig, axes = plt.subplots(figsize=(11,5))
    axes_setup(axes)
    axes.fill_between(Zt.index,
                      (Zt-2.*Zterr).values,
                      (Zt+2.*Zterr).values,
                      facecolor="#0955FE",edgecolor="None",
                      alpha=0.4,zorder=4)
    axes.plot(Zt,color="#0955FE",lw=6,label="Reconstruction",zorder=5)
    
    ## Plot the forecast
    axes.plot(Zt_extrap,color="#FEB209",lw=3,zorder=4,alpha=0.4)
    Zt_extrap.loc[high_seasons != 1] = np.nan
    axes.plot(Zt_extrap,color="#FEB209",lw=6,label="Forecast",zorder=5)
    axes.plot(Zt_extrap-2.*Zterr_extrap,color="#FEB209",ls="dashed",lw=2,zorder=4)
    axes.plot(Zt_extrap+2.*Zterr_extrap,color="#FEB209",ls="dashed",lw=2,zorder=4)
    
    ## Make negative space for timeline stuff
    ylim = axes.get_ylim()
    axes.set_ylim((1.3*ylim[0],1.3*ylim[1]))
    ylim = axes.get_ylim()
    axes.axhline(0,color="grey",ls=":",alpha=0.9,lw=3)

    ## Add cases for context    
    scaled_cases = cases/(cases.max())
    axes.fill_between(cases.index,ylim[0],
                    (0.5*ylim[1]*(scaled_cases.values)+ylim[0]),
                    facecolor="k",edgecolor="None",alpha=0.85,zorder=1)

    ## Add the campaigns
    sias = vax.loc[vax != 0].index
    for d in sias:
        axes.axvline(d,ymin=0,ymax=0.125,lw=3,color="#FE09D0")

    ## Tighten
    axes.set_ylim(ylim)
    axes.set_ylabel("Relative susceptibility")
    fig.tight_layout()
    axes.text(
            0.025,
            0.9,
            countries_list[0],
            fontsize=32,
            color="k",
            transform=axes.transAxes,
            )
    fig.savefig(os.path.join("outputs","forecast_test.png"))
    plt.show()