""" Profiles.py

Regression class and example use case (Figure 2 in the paper) for estimating seasonality
profiles from monthly case time series. """

import sys
import warnings

## Standard imports
import numpy as np
import pandas as pd

## For the log normal cdf, quantiles
from scipy.special import erf, erfinv

## Data processing
from utils import process_case_data

class ProfileRegression:
    def __init__(self, tr_data):
        self.tr_data = tr_data

        ## Make two data frames by country, one for cases today, one
        ## for datas next month
        Cm = tr_data.pivot(index="time", columns="Country", values="Cases")
        Cm_1 = tr_data.pivot(index="time", columns="Country", values="cases_next_month")
        ln_cr = 0.5 * (np.log(Cm_1 + 1) - np.log(Cm + 1))
        self.ln_cr = ln_cr

        ## Make the linear regression operator
        ## Start with the periodic percision matrix
        ## for the prior distribution.
        D2 = (
            np.diag(12 * [-2])
            + np.diag((12 - 1) * [1], k=1)
            + np.diag((12 - 1) * [1], k=-1)
        )
        D2[0, -1] = 1  ## Periodic BCs
        D2[-1, 0] = 1
        pRW2 = np.dot(D2.T, D2) * (
            (2.**4) / 4.0
        )  ## From the total variation of a sine function
        self.pRW2 = pRW2

        ## Then construction the operator mapping beta's to time
        ## stamps
        self.X = np.vstack((int(len(ln_cr) - 1 / len(pRW2)) + 1) * [np.eye(len(pRW2))])[
            : len(ln_cr)
        ]  ## alignment here comes from the start and end months

        ## Then the linear regression operator is
        LR = np.linalg.inv(self.X.T @ self.X + pRW2)

        ## Compute the LR estimates
        self.mu_hat = LR @ self.X.T @ ln_cr.values

        ## Compute the residuals (this is the student's t result)
        RSS = (ln_cr.values - self.X @ self.mu_hat) ** 2
        prior_hat = np.diag(self.mu_hat.T @ pRW2 @ self.mu_hat)
        var = (RSS.sum(axis=0) + prior_hat) / (len(ln_cr) + len(pRW2) - 3)

        ## From which we can compute standard errors
        self.covs = var[:, None, None] * LR[None, :, :]
        self.sigs = np.sqrt(np.diag(LR)[:, None] * var)

        ## Compute the mean and std errors in the effective R
        self.reff = np.exp(self.mu_hat + (self.sigs**2) / 2.0)
        self.reff_err = np.sqrt((np.exp(self.sigs**2) - 1.0)) * self.reff
        self.reff_low = np.exp(self.mu_hat + np.sqrt(2) * self.sigs * erfinv(2 * 0.1 - 1.0))
        self.reff_high = np.exp(self.mu_hat + np.sqrt(2) * self.sigs * erfinv(2 * 0.9 - 1.0))

        ## Compute the low season probabilities
        p_low = 0.5 * (1 + erf((-self.mu_hat) / (self.sigs * np.sqrt(2))))
        self.p_low = p_low

    def periodic_pad(self,a,length=13):
        repeats = int((length/a.shape[0]) + 1)
        return np.vstack(repeats*[a])[:length,:]

if __name__ == "__main__":

    ## Import plotting related libraries and
    ## tools
    import matplotlib.pyplot as plt
    from utils import axes_setup

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

    ## Crayola 8 pack
    c8c = {
        "red": "#ED0A3F",
        "orange": "#FF8833",
        "yellow": "#FBE870",
        "green": "#01A638",
        "blue": "#0066FF",
        "violet": "#803790",
        "brown": "#AF593E",
        "black": "#000000",
    }

    ## Get the data and estimate the profiles
    # Process raw case data
    data = process_case_data(
        "data\\measlescasesbycountrybymonth_Mar2024.csv",
        long_return=True,
        countries_list=["Chad","Ethiopia","Kenya","Nigeria","Pakistan"],
    )
    end_date = "2024-01-01"
    data = data.loc[data["time"] <= end_date]
    logt = ProfileRegression(data)

    ## Plot the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True)
    axes = axes.reshape(-1)
    for ax in axes:
        axes_setup(ax)

    ## Create the trig interpolant design matrix for
    ## periodic interpolation
    tk = np.arange(1.,len(logt.mu_hat)+1.)
    t = np.linspace(1.,len(logt.mu_hat)+1.,395)
    dt = (t[:,None]-tk[None,:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_int = np.sin(np.pi*dt)/(12*np.tan(np.pi*dt/12.))
        X_int = np.nan_to_num(X_int,nan=1.)

    ## plot each country panel
    for i, country in enumerate(logt.ln_cr.columns):

        ## Plot the model with the interpolated
        ## mean and intervals
        mu_hat = X_int @ logt.mu_hat[:,i]
        cov = X_int @ logt.covs[i] @ X_int.T
        sigs = np.sqrt(np.diag(cov)) 
        reff = np.exp(mu_hat + (sigs**2) / 2.0)
        reff_low = np.exp(mu_hat + np.sqrt(2) * sigs * erfinv(2 * 0.1 - 1.0))
        reff_high = np.exp(mu_hat + np.sqrt(2) * sigs * erfinv(2 * 0.9 - 1.0))
        p_low = 0.5 * (1 + erf((-mu_hat) / (sigs * np.sqrt(2))))
        axes[i].fill_between(
            t,
            reff_low,
            reff_high,
            facecolor="k",
            edgecolor="None",
            alpha=0.7,
            zorder=10,
        )
        axes[i].plot(t, reff, color="k", lw=2, alpha=0.9, zorder=20)
             
        ## For a better zoom
        ylim = axes[i].get_ylim()

        ## Plot the data
        hist = data.loc[data["Country"] == country,
                        ["time","Cases"]]\
                        .set_index("time")\
                        .groupby(lambda t: t.month).sum()["Cases"]
        hist = hist/(hist.sum())
        hist = pd.concat([hist,
                          pd.Series([hist.loc[1]],index=[13])],
                          axis=0)
        axes[i].bar(hist.index,4*hist.values+ylim[0],
                    width=1./5.,
                    facecolor="grey",edgecolor="None",
                    alpha=0.6,zorder=0.5)
        
        ## Plot the seasonality classification
        for m, p in enumerate(logt.p_low[:, i]):
            if p >= 0.8:  ## Low season threshold
                color = c8c["green"]
            elif (1 - p) >= 0.8:  ## High season threshold
                color = c8c["red"]
            else:  ## indeterminate
                color = c8c["yellow"]
            axes[i].plot(
                [m + 1.]+ ((m+1) == 1)*[m+13.],
                [logt.reff[m, i]]+((m+1) == 1)*[logt.reff[m, i]],
                marker="o",
                markersize=13,
                ls="None",
                color=color,
                zorder=40,
            )

        ## Some details
        axes[i].axhline(1, color="k", lw=2, zorder=0, ls=":")
        axes[i].set_ylim((ylim[0],ylim[1]))
        axes[i].set_xticks(np.arange(0, 14, 2))
        axes[i].set_xlabel("Month")
        axes[i].set_xticks(np.arange(1,14,2))
        axes[i].set_xticklabels([i%12 for i in np.arange(1,14,2)])
        axes[i].set_ylabel(r"$R_t$")
        axes[i].text(
            0.05,
            0.9,
            country.title(),
            fontsize=26,
            color="k",
            transform=axes[i].transAxes,
        )

    ## Make the legend
    axes[-1].axis("off")
    axes[-1].plot([], color="grey", lw=4, label="Case distribution")
    axes[-1].fill_between([],[], 
        facecolor="k", edgecolor="None", alpha=0.66, label="Model interval")
    axes[-1].plot(
        [], marker="o", markersize=12, ls="None", color=c8c["green"], label="Low season"
    )
    axes[-1].plot(
        [], marker="o", markersize=12, ls="None", color=c8c["yellow"], label="Transition season"
    )
    axes[-1].plot(
        [], marker="o", markersize=12, ls="None", color=c8c["red"], label="High season"
    )
    handles, labels = axes[-1].get_legend_handles_labels()
    order = [0,4,1,2,3]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    axes[-1].legend(handles,labels,frameon=False, loc="center")
    fig.tight_layout()
    fig.savefig("outputs\\seasonality_profiles.png")
    plt.show()

