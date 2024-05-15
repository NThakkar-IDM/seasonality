""" Births.py

Tools to prepare the WoldBank population and crude birth rate datasets."""
import os
import sys

## For analysis and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Import the set of gavi countries
sys.path.append(os.path.join(".."))
from utils.gavi_countries import countries

## Internal helper functions
def get_raw_spreadsheet(root,fname,
                        years=(2010,2019)):

    ## I/O based on pandas with specific dtypes
    ## for speed.
    columns = ["Country Name"]+[str(c) for c in range(*years)]
    dtypes = {c:np.float64 for c in columns[1:]}
    dtypes["Country Name"] = str
    df = pd.read_csv(os.path.join(root,fname),
                     skiprows=4,header=0,
                     usecols=columns,
                     dtype=dtypes)

    ## Adjust country names to the WHO spreadsheet values
    ## done manually here.
    adjustments = {"Yemen, Rep.":"yemen",
                   "Vietnam":"viet nam",
                   "Tanzania":"united republic of tanzania",
                   "Korea, Dem. People’s Rep.":"democratic people's republic of korea",
                   "Gambia, The":"gambia",
                   "Congo, Dem. Rep.":"democratic republic of the congo"}
    df["Country Name"] = df["Country Name"].apply(lambda x: adjustments.get(x,x))

    ## Basic formatting
    df.columns = [c.lower().replace(" name","") for c in df.columns]
    df["country"] = df["country"].str.lower()

    return df

def to_spacetime(df,name):
    df = df.set_index("country").stack(dropna=False).reset_index()
    df.columns = ["country","year",name]
    return df

def GetBirthsSeries(root,
                    years=(2010,2019),
                    countries=None,
                    suffix=""):

    """ Subroutine to get the raw WB spreadsheets (located in root) and to
        reshape it into a space-time dataframe, interpolate, and compute
        births. """

    ## Get the raw data
    population = get_raw_spreadsheet(root,fname="worldbank_totalpopulation"+suffix+".csv",
                                     years=years)
    birthrate = get_raw_spreadsheet(root,fname="worldbank_crudebirthrate"+suffix+".csv",
                                    years=years)

    ## Subset to specific countries
    if countries is not None:
        population = population.loc[population["country"].isin(countries)]
        birthrate = birthrate.loc[birthrate["country"].isin(countries)]

    ## Reshape both into space-time series
    population = to_spacetime(population,"population")
    birthrate = to_spacetime(birthrate,"br")

    ## Convert to timestamps
    population["time"] = pd.to_datetime({"year":population["year"],
                                         "month":6,
                                         "day":15})
    birthrate["time"] = pd.to_datetime({"year":birthrate["year"],
                                        "month":6,
                                        "day":15})

    ## Create a births series
    population = population.set_index(["country","time"])["population"]
    birthrate = birthrate.set_index(["country","time"])["br"]
    births = (population*birthrate/1000).rename("births")
    
    return births

if __name__ == "__main__":

    ## Get the data
    annual_births = GetBirthsSeries(os.path.join("..","..","data"),
                                    countries=countries,
                                    years=(2009,2022),
                                    suffix="_Aug2023")

    ## Interpolate the date after resampling
    ## to the appropriate timescale
    interpolation = lambda s: (s.loc[s.name]/24).resample("SM").interpolate()
    births = annual_births.groupby("country").apply(interpolation)

    ## Serialize the result
    births.to_pickle(os.path.join("..","..","outputs","births.pkl"))
    print(births)   

    ## Plot the interpolation, etc.
    test_country = "chad"
    fig, axes = plt.subplots(figsize=(18,7))
    axes.plot(annual_births.loc[test_country]/24,
              marker="o",ls="None",color="k")
    axes.plot(births.loc[test_country])
    fig.tight_layout()
    plt.show()