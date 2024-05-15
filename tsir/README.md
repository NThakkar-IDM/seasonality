# seasonality\\tsir
The full TSIR model used for comparison purposes in the paper. The model is based on the time series model used in  [*Decreasing measles burden by optimizing campaign timing*](https://www.pnas.org/doi/10.1073/pnas.1818433116), 2019.

There is a required order of operation to reproduce the paper results:
1. All the scripts in `data_analysis/` need to be run (in any order) to generate some serialized pandas outputs for each input.
2. Those outputs need to be compiled into an input dataset, which is done in `PrepareModelingDataset.py`.

Then the remaining scripts can be run in any order. Specifically:
1. `SeasonalityEstimates.py` generates `tsir_profiles.csv` which appears in the paper's third figure.
2. `ScenarioSetCompare.py` generates the endemic average estimates in Figure 5a.
3. `ExtrapolateModel.py` generates the susceptibility estimates and Figure 5b.