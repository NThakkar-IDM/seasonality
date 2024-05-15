""" SIACalendar.py

Source is the WHO spreadsheet found here:
https://www.who.int/immunization/monitoring_surveillance/data/en/ """
import os
import sys
import numpy as np
import pandas as pd

## Country list for dataframe
## construction.
sys.path.append(os.path.join(".."))
from utils.gavi_countries import countries

if __name__ == "__main__":

    ## Process the SIA calendar
    sia_calendar = pd.read_csv(os.path.join("..","..","data","Summary_MR_SIA.csv"),
                          header=1,
                          usecols=["Country","Start date","End date",
                                    "Target population","Reached population"],
                          )
    sia_calendar["time"] = pd.to_datetime(sia_calendar["Start date"]\
                    .str.replace("-","/15/20"),
                    errors="coerce")
    sia_calendar["doses"] = pd.to_numeric(sia_calendar["Reached population"]\
                                    .str.replace(" ",""))
    sia_calendar["doses"] = sia_calendar["doses"].fillna(
        pd.to_numeric(sia_calendar["Target population"]\
                                    .str.replace(" ","")))
    sia_calendar["country"] = sia_calendar["Country"].str.lower()

    ## Get the list of countries
    countries = sorted(list(countries))
    
    ## Estimate target population by dose distribution
    sia_calendar = sia_calendar.loc[sia_calendar["country"].isin(countries)]

    max_doses = sia_calendar[["country","doses"]].groupby("country").max()["doses"]
    sia_calendar["target_pop"] = sia_calendar["doses"]/\
            (max_doses.loc[sia_calendar["country"]].values)
    sia_calendar["target_pop"] = sia_calendar["target_pop"].fillna(0)
    
    ## Create multiindex series with the appropriate shape
    time_index = pd.date_range(start="2008-12-31",
                               end="2023-12-31",
                               freq="SM")
    index = pd.MultiIndex.from_product([countries,time_index],
                                        names=["country","time"])
    target_pop = pd.Series(np.zeros((len(index),)),
                           index=index,
                           name="target_pop")

    ## Loop over SIAs and fill in the data at the
    ## desired admin level
    for i, sia in sia_calendar.iterrows():

        ## Start by getting the closest time to the intended
        ## time index. We use the start date since a lot of the
        ## entries in the dataset have no end date.
        sia_time = time_index[np.argmin(np.abs(time_index-sia.loc["time"]))]

        ## Get the appropriate rows
        rows = [(c, sia_time) for c in countries if c.startswith(sia.loc["country"])]

        ## Set the values
        target_pop.loc[rows] = sia.loc["target_pop"]

    ## Serialize the result
    target_pop.to_pickle(os.path.join("..","..","outputs","target_pop.pkl"))
    print(target_pop.loc[target_pop != 0])