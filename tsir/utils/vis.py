""" vis.py

Plotting and visualization tools. """

## Standards
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## For plotting shapefile polygons
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

##############################################################################################################
## Plotting functions
##############################################################################################################
def VisualizeCountryData(fig,axes,df,
                         colors=["#375E97","#FB6542","#FFBB00"],
                         demography="adj_births",demography_label="RI adjusted births",
                         alt_cases=None):

    """ Simple subroutine to visualize cases, adjusted births, and 
    sias on a two-axis plot. df must have a time index. """

    ## Plot the cases
    if alt_cases is None:
        axes.fill_between(df.index,0,df["cases"].values,
                          alpha=0.5,edgecolor="None",facecolor=colors[0])
    else:
        axes.fill_between(alt_cases.index,0,alt_cases.values,
                          alpha=0.5,edgecolor="None",facecolor=colors[0])

    ## Plot the SIAs
    for x in df.loc[df["target_pop"] != 0].index:
        axes.axvline(x,color=colors[2],ymax=1/6.,alpha=1,lw=1.5)
    axes.plot([],color=colors[2],lw=1.5,label="SIA")

    ## Twin the axes and set up those spines
    axes2 = axes.twinx()

    ## Plot the adj births
    axes2.plot(df[demography],lw=2,color=colors[1])

    ## Details
    axes.set_ylabel("Reported cases",color=colors[0])
    axes2.set_ylabel(demography_label,color=colors[1])
    axes.set_ylim((0,None))
    #axes.ticklabel_format(style="sci",axis="y",scilimits=(0,0))
    #axes2.ticklabel_format(style="sci",axis="y",scilimits=(0,0))   
    fig.tight_layout()

    return

def VisualizeData(fig,axes,df,
                  colors=["#375E97","#FB6542","#FFBB00"],
                  demography="adj_births",demography_label="RI adjusted births",
                  alt_cases=None):

    """ Simple subroutine to visualize cases, adjusted births, and 
    sias on a two-axis plot. df must have a time index. """

    ## Set up the axes
    axes.spines["left"].set_position(("axes",-0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(colors[0])

    ## Plot the cases
    if alt_cases is None:
        axes.fill_between(df.index,0,df["cases"].values,
                          alpha=0.2,edgecolor="None",facecolor=colors[0])
        axes.plot(df["cases"],lw=3,color=colors[0])
    else:
        axes.fill_between(alt_cases.index,0,alt_cases.values,
                          alpha=0.2,edgecolor="None",facecolor=colors[0])
        axes.plot(alt_cases,lw=3,color=colors[0])

    ## Plot the SIAs
    for x in df.loc[df["target_pop"] != 0].index:
        axes.axvline(x,color=colors[2],ymax=1/6.,alpha=1,lw=3)
    axes.plot([],color=colors[2],lw=3,label="Historical vaccination campaigns")

    ## Twin the axes and set up those spines
    axes2 = axes.twinx()
    axes2.spines["right"].set_position(("axes",1.025))
    axes2.spines["top"].set_visible(False)
    axes2.spines["left"].set_visible(False)
    axes2.spines["bottom"].set_visible(False)
    axes2.spines["right"].set_color(colors[1])

    ## Plot the adj births
    axes2.plot(100*df["mcv1"],lw=3,color=colors[1])

    ## Details
    axes.set_ylabel("Reported measles cases",color=colors[0],labelpad=15)
    axes2.set_ylabel(demography_label,color=colors[1],labelpad=15)
    axes.set_ylim((0,None))
    #axes2.set_ylim((None,74))
    axes.tick_params(axis="y",colors=colors[0])
    axes2.tick_params(axis="y",colors=colors[1])
    fig.tight_layout()

    return

## Helper functions
def low_mid_high(samples):
    low = np.percentile(samples,2.5,axis=0)
    mid = np.mean(samples,axis=0)
    high = np.percentile(samples,97.5,axis=0)
    return low,mid,high