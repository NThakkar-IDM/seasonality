""" PrepareModelingDataset.py

Script to take serialized timeseries from data_analysis\\, combine them, fill NaNs,
and visualize data from particular countries. """

import os

## Standard imports
import pandas as pd
import matplotlib.pyplot as plt

## Visualization tools
from utils.vis import *

def GetCombinedDataset(serial_box=None,fillna=True):

    """ Function to concetate timeseries and make a single space-time
    dataframe with filled NaNs. 
    
    serial_box = {column_name:pickle_path,...} and it must contain keys
    "cases","mcv1", "target_pop", and "births". """

    if serial_box is None:
        serial_box = {"cases":os.path.join("..","outputs","epi_curves.pkl"),
                      "mcv1":os.path.join("..","outputs","mcv1.pkl"),
                      "mcv2":os.path.join("..","outputs","mcv2.pkl"),
                      "births":os.path.join("..","outputs","births.pkl"),
                      "target_pop":os.path.join("..","outputs","target_pop.pkl")}
    else:
        assert set(serial_box.keys()) >= {"cases","mcv1","target_pop","births"},\
               "Must specify pickles for cases, mcv1, and births."

    ## Use the filepaths to create a combined dataframe
    df = pd.concat([pd.read_pickle(v).rename(k) for k,v in serial_box.items()],
                   axis=1)

    ## Compute RI adjusted births
    if "mcv2" in df.columns:
        df["adj_births"] = df["births"]*(1.-0.9*df["mcv1"]*(1.-df["mcv2"])\
                                         -0.99*df["mcv1"]*df["mcv2"])
    else:
        df["adj_births"] = df["births"]*(1.-0.9*df["mcv1"])

    if fillna:
        df = df.loc[df["cases"].notnull()]
        df["adj_births"] = df["adj_births"].groupby("country").apply(\
                                            lambda s: s.fillna(method="ffill").fillna(method="bfill"))

    return df[["cases","adj_births","target_pop","mcv1","mcv2","births"]]

if __name__ == "__main__":

    ## Get timeseries from data_analysis, compile and
    ## serialize
    df = GetCombinedDataset()
    df = df.loc(axis=0)[:,:"2023-12-31"]
    df.to_pickle(os.path.join("..","outputs","modeling_dataset.pkl"))
    print(df)

    ## Make a simple plot
    fig, axes = plt.subplots(figsize=(12,5))
    VisualizeData(fig,axes,df.loc["chad"],
                  demography="mcv1",demography_label="MCV1 coverage",
                  )
    axes.legend(loc=1,frameon=False)
    plt.show()



