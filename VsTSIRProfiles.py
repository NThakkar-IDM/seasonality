""" VsTSIRProfiles.py


Comparing the regression method to TSIR outputs (saved to a csv, via the public data
workflow associated with Thakkar et al., PNAS 2019). """

import os

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Data processing
from Profiles import ProfileRegression
from utils import process_case_data, axes_setup

# Diagnostic plot
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

if __name__ == "__main__":

    ## Get the TSIR profile estimates
    tsir_df = pd.read_csv(os.path.join("data","tsir_profiles.csv"),
        index_col=0,
        parse_dates=["time"],
        date_parser=pd.to_datetime,
        )
    tsir_df["country"] = tsir_df["country"].str.title()
    countries_list = list(tsir_df["country"]
                    .unique())
    tsir_df = tsir_df.set_index(["country","time"])
    tsir_df = tsir_df.groupby("country").apply(
                lambda s: s.loc[s.name].resample("M").mean()
                )

    ## Process the data and regress
    data = process_case_data(
        os.path.join("data","measlescasesbycountrybymonth_Mar2024.csv"),
        long_return=True,
        countries_list=countries_list,
    )
    end_date = "2024-01-01"
    data = data.loc[data["time"] <= end_date]
    logt = ProfileRegression(data)

    ## Set up a scatter plot
    scatter_fig, scatter_axes = plt.subplots(figsize=(8,7))
    axes_setup(scatter_axes)
    scatter_axes.grid(color="grey",alpha=0.2)

    ## Loop over countries and add
    for i, country in enumerate(logt.ln_cr.columns):

        ## Plot the scatter plot
        this_tsir = tsir_df.loc[country]
        scatter_axes.errorbar(logt.reff[:, i],
                            this_tsir["rel_reff"].values,
                            xerr=logt.reff_err[:,i],
                            yerr=this_tsir["rel_reff_err"].values,
                            ls="None",marker="o",markersize=12,color="k",zorder=12)

    ## Add the details
    scatter_axes.set_ylabel(r"Rel. $\beta_t$ in the transmission model") 
    scatter_axes.set_xlabel(r"R$_t$ regression model")

    ## Make a simple regression line
    x = logt.reff.T.reshape((len(tsir_df),))
    X = np.array([np.ones((len(x),)),
                  x]).T
    Y = tsir_df.loc[sorted(countries_list),"rel_reff"].values
    L = np.linalg.inv(X.T @ X)
    beta_hat = L @ X.T @ Y
    rss = np.sum((Y - X @ beta_hat)**2)
    beta_cov = rss*L/(len(Y)-2)
    X_fine = np.ones((100,2))
    X_fine[:,1] = np.linspace(0.9*x.min(),
                    1.1*x.max(),
                    X_fine.shape[0])
    Yhat = X_fine @ beta_hat
    Ycov = X_fine @ beta_cov @ X_fine.T
    Ystd = np.sqrt(np.diag(Ycov))
    scatter_axes.plot(X_fine[:,1],Yhat,color="xkcd:saffron",lw=5,ls="dashed",zorder=9)
    corr = np.corrcoef(x,Y)[0,1]
    print("Correlation coefficient = {}".format(corr))
   
    ## Finish up
    scatter_fig.tight_layout()
    scatter_fig.savefig(os.path.join("outputs","vs_tsir_scatter.png"))
    plt.show()
