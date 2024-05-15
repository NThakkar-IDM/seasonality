""" RI.py

Tools to prepare RI timeseries using the WUENIC estimates for each country. """
import os
import sys

## For analysis and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Import the set of gavi countries
sys.path.append(os.path.join(".."))
from utils.gavi_countries import countries

def get_raw_spreadsheet(vaccine,root,
                        fname="coverage_estimates_series.xls",
                        years=(2010,2019)):

    ## Updated WHEUNIC estimates have a new format, so we have
    ## two different I/O patterns implemente
    dtypes = {"NAME":str,
              "YEAR":str,
              "ANTIGEN":str,
              "COVERAGE_CATEGORY_DESCRIPTION":str,
              "COVERAGE":np.float64}
    columns = list(dtypes.keys())
    df = pd.read_csv(os.path.join(root,fname),
                       header=0,
                       usecols=columns,
                       dtype=dtypes,
                       )

    ## Subset and reshape
    df = df.loc[(df["ANTIGEN"] == vaccine) & \
                (df["COVERAGE_CATEGORY_DESCRIPTION"] == \
                "WHO/UNICEF Estimates of National Immunization Coverage")]
    df["YEAR"] = df["YEAR"].astype(np.int32)

    ## Adjust country names to the WHO spreadsheet values
    ## done manually here.
    adjustments = {"Democratic People's Republic of Korea (the)":"democratic people's republic of korea",
                   "Democratic Republic of the Congo (the)":"democratic republic of the congo",
                   "occupied Palestinian territory, including east Jerusalem":"palestine"}
    df["NAME"] = df["NAME"].apply(lambda x: adjustments.get(x,x))

    ## Pivot into the data structure for interpolation
    df = df[["NAME","YEAR","COVERAGE"]].pivot(index="NAME",
                                              columns="YEAR",
                                              values="COVERAGE")
    df = df.reset_index().rename(columns={"NAME":"country"})

    ## Subset to relevant years, do some simple formating
    df = df[["country"]+list(range(*years))].copy()
    df["country"] = df["country"].str.lower()
    df.columns.name = None
        
    return df

def GetCoverageSeries(vaccine,root,
                      fname="coverage_estimates_series.xls",
                      years=(2010,2019),
                      countries=None):

    """ Subroutine to create coverage timeseries. Vaccine corresponds to
    a sheet in the WUENIC dataset. """

    ## Get the raw data
    df = get_raw_spreadsheet(vaccine,root,fname,years)
    
    ## Subset to specific countries
    if countries is not None:
        df = df.loc[df["country"].isin(countries)]

    ## Reshape to spacetime df
    df = df.set_index("country").stack(dropna=False).reset_index()
    df.columns = ["country","year",vaccine.lower()]

    ## Convert to timestamps
    df["time"] = pd.to_datetime({"year":df["year"],
                                 "month":6,
                                 "day":15})

    ## Reshape as a timeseries and scale
    df = df.set_index(["country","time"])[vaccine.lower()]/100.

    return df


if __name__ == "__main__":

    for coverage_name in ["MCV1", "MCV2"]:

        ## Get the data
        annual_coverage = GetCoverageSeries(coverage_name,os.path.join("..","..","data"),
                                            fname="coverage_estimates_Aug2023.csv",
                                            countries=countries,
                                            years=(2009,2023))
        
        ## NaNs here are places without vaccine (MCV2) introduction
        annual_coverage = annual_coverage.fillna(0)

        ## Interpolate the date after resampling
        ## to the appropriate timescale
        interpolation = lambda s: s.loc[s.name].resample("SM").interpolate()
        coverage = annual_coverage.groupby("country").apply(interpolation)

        ## Serialize the result
        coverage.to_pickle(os.path.join("..","..","outputs","{}.pkl".format(coverage.name)))

        ## Plot the interpolation, etc.
        test_country = "chad"
        fig, axes = plt.subplots(figsize=(18,7))
        axes.plot(annual_coverage.loc[test_country],
                marker="o",ls="None",color="k")
        axes.plot(coverage.loc[test_country])
        fig.tight_layout()
        plt.show()