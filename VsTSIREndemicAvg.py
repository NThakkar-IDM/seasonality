""" VsTSIREndemicAvg.py

Comparing seasonality-based endemic flucations with long-time TSIR outputs. """
import os
import sys

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Data processing
from Profiles import ProfileRegression
from utils import process_case_data,\
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

if __name__ == "__main__":

    ## countries (plotting built for only 4)
    countries_list = ["Chad","Kenya"]

    # Process raw case data
    data = process_case_data(
        os.path.join("data","measlescasesbycountrybymonth_Mar2024.csv"),
        long_return=True,
        countries_list=countries_list,
    )
    end_date = "2024-01-01"
    data = data.loc[data["time"] <= end_date]

    ## Fit the profile model
    logt = ProfileRegression(data)
    i_to_c = {i:c for i,c in enumerate(logt.ln_cr.columns)}
    c_to_i = {c:i for i,c in i_to_c.items()}

    ## Set up a figure
    fig, axes = plt.subplots(2,1,figsize=(8,10),sharex=True,sharey=False)
    axes = axes.reshape(-1)
    for ax in axes:
        axes_setup(ax)
        ax.set_zorder(2)
        ax.patch.set_facecolor("None")

    for i, country in enumerate(countries_list):

        ## Get the model outputs
        tsir_df = pd.read_csv(os.path.join("data","{}_sia_comparisons.csv".format(
                    countries_list[c_to_i[country]].lower()
                    )),
                index_col=0,
                parse_dates=["sia_date"],
                date_parser=pd.to_datetime,
                )

        ## Compute demographic pressure parameters via
        ## endemic stability
        t = np.arange(len(logt.mu_hat))
        stability_extent = 3 #int((len(data)/12 + 1))
        A0 = np.tril(np.ones((len(t),len(t))),k=-1) 
        gf = np.exp((A0 @ (2.*logt.mu_hat[:,c_to_i[country]]))\
                + 0.5*np.diag(A0 @ (4.*logt.covs[c_to_i[country]]) @ A0.T))
        #gf_std = gf * np.sqrt(np.exp(4*np.diag(A0 @ logt.covs[c_to_i[country]] @ A0.T))-1.)
        gf = np.hstack(stability_extent*[gf])
        #gf_std = np.hstack(stability_extent*[gf_std])
        A1 = np.tril(np.ones((len(gf),len(gf))),k=0)
        A1[:,0] = 0 
        X = np.array([np.ones((len(gf),)),
                      np.arange(len(gf))]).T
        L = np.linalg.inv(X.T @ X) @ X.T 
        #resid_op = np.eye(len(X)) - X @ L
        alphas = L @ A1 @ gf
        rel_S = (X @ alphas) - (A1 @ gf)

        ## Make the cartoon plot for the 
        ## first country.
        if i == 0:
            fig2, axes2 = plt.subplots(figsize=(8,7))
            axes_setup(axes2)
            axes2.grid(color="grey",alpha=0.2)
            axes2.spines["left"].set_visible(False)
            axes2.plot(-A1 @ gf,color="k",lw=5,label="Endemic burden")
            axes2.plot(X @ alphas,color="grey",lw=5,ls="-.",label="Implied demographic\npressure")
            axes2.plot(rel_S,color="xkcd:saffron",lw=6,label="Endemic relative\nsusceptibility")
            axes2.set_yticks([])
            axes2.legend(frameon=False,loc=2,fontsize=22)
            axes2.set_xlabel("Time (months)")
            ylim = axes2.get_ylim()
            axes2.set_ylim((1.1*ylim[0],1.3*ylim[1]))
            fig2.tight_layout()
            fig2.savefig(os.path.join("outputs","endemic_stability.png"))

        ## Plot relative susceptibiplity 
        t_avg_S = tsir_df["avg_S"].mean()

        ## Plot it
        axes[i].errorbar(np.arange(1,len(tsir_df)+1),
                    tsir_df["avg_S"]/t_avg_S - 1.,
                    yerr=2.*tsir_df["std_S"]/t_avg_S,
                    marker="o",ls="None",
                    color="k",markersize=0,alpha=0.3,zorder=1)
        im = axes[i].scatter(np.arange(1,len(tsir_df)+1),
                   tsir_df["avg_S"]/t_avg_S - 1.,
                   marker="o",s=15**2,
                   c="k",
                   zorder=2
                   )

        ## Add the model result
        y = (tsir_df["avg_S"]/t_avg_S).values
        sf = ((y-1)/rel_S[:len(y)])[0]
        axes[i].plot(np.arange(1,len(y)+1),sf*rel_S[:len(y)],color="xkcd:saffron",lw=4)
    
        ## Label some stuff
        axes[i].set_ylabel("Rel. susceptibility")
        axes[i].text(0.025,0.975,countries_list[c_to_i[country]],
                fontsize=28,color="k",
                horizontalalignment="left",verticalalignment="top",
                transform=axes[i].transAxes)
        ylim = axes[i].get_ylim()
        axes[i].set_ylim((1.1*ylim[0],1.3*ylim[1]))  

    ## Details and adjustments for the 
    ## zoomed in inset
    axes[-1].set_xlabel("Time of year (month)")
    fig.tight_layout()
    fig.subplots_adjust(right=0.7,
                        #bottom=0.2,
                        )

    ## Add the inset zoom
    for i, country in enumerate(countries_list):

        ## Get the model outputs
        tsir_df = pd.read_csv(os.path.join("data","{}_sia_comparisons.csv".format(
                    countries_list[c_to_i[country]].lower()
                    )),
                index_col=0,
                parse_dates=["sia_date"],
                date_parser=pd.to_datetime,
                )

        ## Compute demographic pressure parameters via
        ## endemic stability
        t = np.arange(len(logt.mu_hat))
        A0 = np.tril(np.ones((len(t),len(t))),k=-1) 
        gf = np.exp((A0 @ (2.*logt.mu_hat[:,c_to_i[country]]))\
                + 0.5*np.diag(A0 @ (4.*logt.covs[c_to_i[country]]) @ A0.T))
        gf = np.hstack(stability_extent*[gf])
        A1 = np.tril(np.ones((len(gf),len(gf))),k=0)
        A1[:,0] = 0 
        X = np.array([np.ones((len(gf),)),
                      np.arange(len(gf))]).T
        L = np.linalg.inv(X.T @ X) @ X.T 
        alphas = L @ A1 @ gf
        rel_S = (X @ alphas) - (A1 @ gf)

        ## Add the zoom inset
        x1, x2 = axes[i].get_xlim()
        x1 = 1.01*x1
        x2 = 0.99*x2
        y1 = 0.965 - 1.
        y2 = 1.035 - 1.
        t_avg_S = tsir_df["avg_S"].mean()
        mod_ax = axes[i].inset_axes([0.85, -0.2, 0.65, 0.6], #[0.49, 0.22, 0.26, 0.85],
                xlim=(x1, x2), ylim=(y1, y2), xticks=[], yticks=[],
                zorder=10+i)
        mod_ax.errorbar(np.arange(1,len(tsir_df)+1),
                    tsir_df["avg_S"]/t_avg_S - 1.,
                    yerr=2.*tsir_df["std_S"]/t_avg_S,
                    marker="o",ls="None",
                    color="k",markersize=0,alpha=0.1,zorder=1)
        im = mod_ax.scatter(np.arange(1,len(tsir_df)+1),
                   tsir_df["avg_S"]/t_avg_S - 1.,
                   marker="o",s=15**2,
                   c="k",
                   zorder=2,
                   )
        y = (tsir_df["avg_S"]/t_avg_S).values
        sf = ((y-1)/rel_S[:len(y)])[0]
        mod_ax.plot(np.arange(1,len(y)+1),sf*rel_S[:len(y)],color="xkcd:saffron",lw=4)
        box, lines = axes[i].indicate_inset_zoom(mod_ax,edgecolor="k",lw=1)

    ## Finish up
    fig.savefig(os.path.join("outputs","vs_endemic_2country.png"))
    plt.show()
    