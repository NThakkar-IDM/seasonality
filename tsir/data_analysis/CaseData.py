""" CaseData.py

Tools  to work with the WHO spreadsheet of case data by country and by month. """
import os
import sys

## For analysis and I/O
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Import the set of gavi countries
sys.path.append(os.path.join(".."))
from utils.gavi_countries import countries

## Internal helper functions
def get_raw_spreadsheet(root,
                        fname="measlescasesbycountrybymonth_Mar2024.csv"):

    ## I/O based on pandas with specific dtypes
    ## for speed.
    columns = ["Country","Year",
               "January","February","March","April",
               "May","June","July","August","September",
               "October","November","December"]
    dtypes = {c:np.float64 for c in columns[2:]}
    dtypes["Country"] = str
    dtypes["Year"] = np.int32

    ## Get the CSV
    df = pd.read_csv(os.path.join(root,fname),
                    header=0,
                    usecols=columns,
                    dtype=dtypes)

    ## Basic formatting
    df.columns = [c.lower() for c in df.columns]
    df["country"] = df["country"].str.lower()

    return df

def semi_monthly_resample(df):
    resample_func = lambda s: s.loc[s.name].resample("SM").asfreq()
    return df.groupby("country").apply(resample_func)

def semi_monthly_smooth(df,w=3):
    smooth_func = lambda s: (s.loc[s.name]/2.).resample("SM").bfill().rolling(w).mean()
    return df.groupby("country").apply(smooth_func)

## Subroutine for data retrieval and processing.
def GetEpiCurveSeries(root,
                      fname="measlescasesbycountrybymonth_Mar2024.csv",
                      countries=None,
                      sm_smooth=True,sm_resample=False):

    """ Subroutine to get the raw WHO spreadsheet (at path root+fname) and to
        reshape it into a space-time dataframe. """

    ## Get the raw data
    df = get_raw_spreadsheet(root,fname)

    ## Subset to country set if specified
    if countries is not None:
        df = df.loc[df["country"].isin(countries)]

    ## Reshape so the individual month columns are stacked
    ## into a single column
    df = df.set_index(["country","year"]).stack(dropna=True).reset_index()
    df.columns = ["country","year","month","cases"]

    ## Create a time series for each row, with date set to the
    ## end of the month.
    df["time"] = pd.to_datetime(df["year"].astype(str)+df["month"],
                                format="%Y%B")
    df["time"] = df["time"].dt.to_period("M").dt.to_timestamp("M")

    ## Finally, clean up
    df = df[["country","time","cases"]].set_index(["country","time"])["cases"]

    ## And if needed, resample and smooth
    if sm_smooth:
        df = semi_monthly_smooth(df)
    elif sm_resample:
        df = semi_monthly_resample(df)

    return df

if __name__ == "__main__":

    ## Get the uninterpolated data
    df = GetEpiCurveSeries(os.path.join("..","..","data"),
                           countries=countries,
                           sm_smooth=False,sm_resample=False)
    print(df)

    ## As a test
    test_country = "chad"
    s = df.loc[test_country]
    smoothed = GetEpiCurveSeries(os.path.join("..","..","data"),
                                countries=countries)
    print(smoothed)

    ## Serialize the results
    df.to_pickle(os.path.join("..","..","outputs","raw_cases.pkl"))
    smoothed.to_pickle(os.path.join("..","..","outputs","epi_curves.pkl"))

    ## Test plot
    fig, axes = plt.subplots(figsize=(18,7))
    axes.plot(s/2,marker="o",ls="None")
    axes.plot(smoothed.loc[test_country])
    fig.tight_layout()
    plt.show()

