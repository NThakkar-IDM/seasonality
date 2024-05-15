""" SeasonalityEstimates.py

Computing some seaonality profiles for comparison to the regression method. """
import os
import sys

## Standard imports
import numpy as np
import pandas as pd

## TSIR model functions
import utils.tsir as tsir

if __name__ == "__main__":

    ## Get the dataset
    countries = ["kenya","chad","ethiopia","nigeria","pakistan"]
    dataset = pd.read_pickle(os.path.join("..","outputs","modeling_dataset.pkl"))
    df = dataset.loc(axis=0)[countries,"2012-01-01":]
    
    ## Print the dataset
    print("\nDataset for analysis:")
    print(df)

    ## Fit the model. This is done with a basic, 
    ## i.e. non-spatial, TSIR model.
    print("\nLooping over countries...")
    seasonality_profiles = {}
    for country in countries:
        sf = df.loc[country].copy()
        print("Fitting the model for {}...".format(country.title()))
        sf, model, rr = tsir.FitTSIRModel(sf,
                                         tsir.BasicSusceptibleReconstruction,
                                         tsir.BasicTransmissionRegression,
                                         verbose=False)
        theta = model["params"]
        theta_cov = model["params_var"]
        tau = model["periodicity"]
        reff = np.exp(theta[:tau] + 0.5*np.diag(theta_cov)[:tau])
        reff_err = np.sqrt(np.exp(np.diag(theta_cov)[:tau])-1)*reff
        reff = reff - reff.mean()

        ## Save the result
        sm_dates = sf.index[:model["periodicity"]]
        sm_dates = pd.to_datetime({"year":sm_dates[0].year,
                                    "day":sm_dates.day,
                                    "month":sm_dates.month}).sort_values()
        profile = pd.DataFrame(np.array([reff,reff_err]).T,
                        index=np.roll(sm_dates,1),
                        columns=["rel_reff","rel_reff_err"])
        seasonality_profiles[country] = profile

    ## Reshape
    seasonality_profiles = pd.concat(seasonality_profiles.values(),
                                     keys=seasonality_profiles.keys())
    seasonality_profiles = seasonality_profiles.reset_index()
    seasonality_profiles.columns = ["country","time","rel_reff","rel_reff_err"]
    print("\nFinal output:")
    print(seasonality_profiles)
    seasonality_profiles.to_csv(os.path.join("..","data","tsir_profiles.csv"))