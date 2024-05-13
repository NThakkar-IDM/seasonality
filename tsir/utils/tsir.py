""" tsir.py

Functions and tools to facilitate TSIR Model fitting and sampling."""
import sys
sys.path.insert(0,"..\\..\\")

## Standard imports 
import numpy as np
import pandas as pd

## For R2 scores
from sklearn.metrics import r2_score

## For optimization
from scipy.optimize import minimize

## Helper functions
def up_sample(x):
    total_pop = x.population.sum()
    target_pop = np.sum(x.population*x.target_pop)/total_pop
    series = pd.Series([x.adj_births.sum(),x.cases.sum(),target_pop],
                      index=["adj_births","cases","target_pop"])
    return series

def WeightedLeastSquares(X,Y,weights=None,verbose=False,standardize=True):

    """ Weighted LS, reduces to OLS when weights is None. This implementation computes
    the estimator and covariance matrix based on sample variance. For TSIR however, the

    NB: x is assumed to be an array with shape = (num_data points, num_features) """

    ## Get the dimensions
    num_data_points = X.shape[0]
    try:
        num_features = X.shape[1]
    except:
        num_features = 1
        X = X.reshape((num_data_points,1))

    ## Initialize weight matrix
    if weights is None:
        W = np.eye(num_data_points)
    else:
        W = np.diag(weights)

    ## Standardize the inputs and outputs to help with
    ## stability of the matrix inversion. This is needed because
    ## cumulative cases and births both get very large.
    if standardize:
        muY = Y.mean()
        sigY = Y.std()
        muX = X.mean(axis=0)
        sigX = X.std(axis=0)
        X = (X-muX)/sigX
        Y = (Y-muY)/sigY

    ## Compute the required matrix inversion
    ## i.e. inv(x.T*w*x), which comes from minimizing
    ## the residual sum of squares (RSS) and solving for
    ## the optimum coefficients. See eq. 3.6 in EST
    xTwx_inv = np.linalg.inv(np.dot(X.T,np.dot(W,X)))

    ## Now use that matrix to compute the optimum coefficients
    ## and their uncertainty.
    beta_hat = np.dot(xTwx_inv,np.dot(X.T,np.dot(W,Y)))

    ## Compute the estimated variance in the data points
    residual = Y - np.dot(X,beta_hat)
    RSS = (residual)**2
    var = RSS.sum(axis=0)/(num_data_points - num_features)

    ## Then the uncertainty (covariance matrix) is simply a 
    ## reapplication of the inv(x.T*x):
    beta_var = var*xTwx_inv

    ## Reshape the outputs
    beta_hat = beta_hat.reshape((num_features,))

    ## Rescale back to old values
    if standardize:
        X = sigX*X + muX
        Y = sigY*Y + muY
        beta_hat = beta_hat*(sigY/sigX)
        sig = np.diag(sigY/sigX)
        beta_var = np.dot(sig,np.dot(beta_var,sig))
        residual = sigY*residual + muY - np.dot(muX,beta_hat)

    ## Print summary if needed
    if verbose:
        for i in range(num_features):
            output = (i,beta_hat[i],2.*np.sqrt(beta_var[i,i]))
            print("Feature %i: coeff = %.4f +/- %.3f." % output)

    return beta_hat, beta_var, residual

###############################################################################################################
#### TSIR model fitting functions
###############################################################################################################
def BasicSusceptibleReconstruction(df,sia,rep_rate_uq=False):

    """ S_t reconstruction and reporting rate estimation assuming some SIAs. df must contain columns adj_births,
    cases, and target_pop. sia is a series with non-zero entries. Output is the reporting rate, Z_t, and 
    I_t estimates."""

    ## Make sure the SIA column has non-zero entries. If it doesn't, we
    ## default to the old method.
    if len(sia[sia != 0]) == 0:
    
        ## Compute the features and response
        response = np.cumsum(df.adj_births.values+1.)
        features = np.cumsum(df.cases.values+1.).reshape(-1,1)

        ## Construct the weights. Weights are based on the variance
        ## of [I_t | C_t, p].
        weights = 1./np.sqrt(df.cases.values + 1.)

        ## Compute the MLE
        beta, beta_var, Z_t = WeightedLeastSquares(features,response,weights,verbose=False)

        ## Compute high level results
        reporting_rate = 1./beta[0] 
        I_t = beta[0]*(df.cases.values+1.)-1.

        return reporting_rate, Z_t, I_t

    ## Construct the SIA matrix, a (t x t) matrix
    ## with 1-sia coverage cumulative products as entries.
    n_steps = len(df)
    M_sia = np.zeros((n_steps-1,n_steps-1))
    sia_c = 1. - sia.values
    for i in range(n_steps-1):
        M_sia[i:,i] = np.cumprod(sia_c[i:-1])

    ## Now construct the adjustment matrices.   
    ## D (for the intercept) and A (for the births and cases)
    ## For D, we need a column of zeros and then the sia block.
    D = np.zeros((n_steps,n_steps))
    D[1:,1:] = M_sia

    ## A has a repeated M_sia column as the first column, 
    ## with a 1 in the extra entry. (See the notes for details).
    A = np.eye(n_steps)
    A[1:,0] = M_sia[:,0]
    A[1:,1:] = M_sia

    ## Now we create the output vector
    output = np.dot(A,df["adj_births"].values+1.)

    ## And the feature matrix. x0 is the sia-adjusted cumulative cases,
    ## which corresponds to the reporting rate and x1
    ## corresponds to the intercept (which is identifiable only due to
    ## sia's).
    x0 = np.dot(A,(df["cases"].values+1.).reshape(-1,1))
    x1 = np.zeros((n_steps,1)) 
    x1[1:,0] = sia.values[:-1]/sia_c[:-1]
    x1 = np.dot(D,x1)

    ## Since we're using the detrended method, S_t = S_bar + Z_t where
    ## S_bar is a constant.
    features = np.concatenate([x0,x1],axis=1)

    ## Construct the weights. weights are based on the Bayesian approach's
    ## variance in the I_t | C_t distribution
    weights = 1./np.sqrt(df["cases"].values + 1.)

    ## Compute the MLE
    beta, beta_var, Z_t = WeightedLeastSquares(features,output,weights)

    ## Compute high level results
    reporting_rate = 1./beta[0] 
    I_t = beta[0]*(df["cases"].values+1.)-1.

    ## If we really care about the reporting rate and its
    ## uncertainty.
    if rep_rate_uq:
        sig2_rho = beta_var[0,0]
        sig2_rep_rate = sig2_rho/(beta[0]**4)
        return reporting_rate, sig2_rep_rate

    return reporting_rate, Z_t, I_t

def BasicTransmissionRegression(df,Z_t,I_t,periodicity=24):

    """ Transmission regression without any spatial information, to be used for I_t and Z_t inference in the
    neighborhood model. SIAs create the need for this - we have to iterate over model fits in the neighborhood to
    properly account for SIAs and get a good estimate of infections and susceptibility there. 

    df here is the df containing adj_births, cases, and SIAs for the region being modelled. """

    ## Set up the basic feature matrix and response vector, Nxp and (N,) respectively.
    ## These are the same as the basic TSIR model, found in the tsir.py class.
    N = len(df)-1
    p = periodicity+2
    X = np.zeros((N,p))
    Y = np.log(I_t[1:])
    X[:,:periodicity] = np.vstack((int(N/periodicity)+1)*[np.eye(periodicity)])[1:N+1]
    X[:,periodicity] = np.log(I_t[:-1])
    X[:,periodicity+1] = Z_t[:-1]

    ## Compute the regression
    params, params_var, residual = WeightedLeastSquares(X,Y,standardize=False)

    ## Compute the sample variance
    RSS = (residual)**2
    var = RSS.sum(axis=0)/(N-p)

    ## Compute the standard error in the betas via taylor series
    S_bar = 1./params[periodicity+1]
    sig2s = np.diag(params_var)
    sig2 = sig2s[:periodicity] + sig2s[periodicity+1]/(S_bar**2)
    t_sig = np.exp(params[:periodicity])*np.sqrt(sig2)/S_bar

    ## Compute some highlevel things we need for model testing, sampling, etc
    transmission_model = {"params":params,
                          "params_var":params_var,
                          "S_bar":1./params[periodicity+1],
                          "S_bar_std":np.sqrt(params_var[periodicity+1,periodicity+1])/(params[periodicity+1]**2),
                          "t_beta":np.exp(params[:periodicity])*params[periodicity+1],
                          "t_beta_sig":t_sig,
                          "alpha":params[periodicity],
                          "alpha_std":np.sqrt(np.diag(params_var)[periodicity]),
                          "std_logE":np.sqrt(var),
                          "scale_factor": 1.,
                          "periodicity":periodicity}

    return transmission_model

def LongTermR2Score(theta,df,
                    susceptible_reconstruction_function,transmission_regression_function,
                    mse=False,cutoff=0):

    """ Function to compute a TSIR model skeleton given vector theta = [mu_1,...,mu_n] where
    n is the number of SIAs (i.e. len(target_pop_fraction.loc[!= 0.]))

    Function outputs R2_values to be optimized numerically. """

    ## Create the SIA column by setting non-zero entries of the
    ## df's target population column. Total SIA efficacy = efficacy*target_pop
    sia_efficacies = theta*df.loc[df["target_pop"] != 0.,"target_pop"].values
    sia = df["target_pop"].rename("sia")
    sia.loc[sia != 0.] = sia_efficacies

    ## Fit the model with these SIA efficacies
    reporting_rate, Z_t, I_t = susceptible_reconstruction_function(df,sia)
    transmission_model = transmission_regression_function(df,Z_t,I_t)

    ## Compute the skeleton and inferred I
    skeleton = np.zeros((len(df),))
    skeleton[0] = I_t[0]
    S_skeleton = np.zeros((len(df),))
    S_skeleton[0] = transmission_model["S_bar"] + Z_t[0]

    ## Loop over time and compute the skeleton
    for i in range(1,len(df)):
        time_in_period = i % transmission_model["periodicity"]
        skeleton[i] = transmission_model["scale_factor"]*transmission_model["t_beta"][time_in_period]*\
                      (S_skeleton[i-1])*(skeleton[i-1]**transmission_model["alpha"])
        S_skeleton[i] = (S_skeleton[i-1] + df["adj_births"].values[i] - skeleton[i])*(1.-sia.values[i-1])

    ## Compute R2 score
    if mse:
        return -0.5*np.sum((I_t[cutoff:] - skeleton[cutoff:])**2) 
    else:
        return r2_score(I_t[cutoff:],skeleton[cutoff:])

def FitTSIRModel(df,susceptible_reconstruction_function,transmission_regression_function,
                 initial_guess=0.25,cutoff=0,verbose=True):

    ## Set up the initial guess
    num_params = len(df.loc[df["target_pop"] != 0.])
    x0 = initial_guess*np.ones((num_params,))

    ## Use scipy.minimize on -R2Score
    f = lambda x: -LongTermR2Score(x,df,
                                   susceptible_reconstruction_function,
                                   transmission_regression_function,
                                   mse=False,cutoff=cutoff)
    result = minimize(f,x0,method="L-BFGS-B",
                      bounds=num_params*[(0.,0.999)])

    ## Summarize the optimization results
    if verbose:
        if not result["success"]:
            print("\nModel fitting failed!")
            print(result)
        else:
            print("Final model performance = {}".format(-result["fun"]))

    ## Create the SIA column by setting non-zero entries of the
    ## df's target population column. Total SIA efficacy = efficacy*target_pop
    sia_efficacies = result["x"]*df.loc[df["target_pop"] != 0.,"target_pop"].values
    sia = df["target_pop"].rename("sia")
    sia.loc[sia != 0.] = sia_efficacies

    ## Fit the model with these SIA efficacies
    reporting_rate, Z_t, I_t = susceptible_reconstruction_function(df,sia)
    transmission_model = transmission_regression_function(df,Z_t,I_t)

    ## Store the end results
    df["sia"] = sia
    df["Z_t"] = Z_t
    df["I_t"] = I_t

    return df, transmission_model, reporting_rate

def SampleBasicTSIR(df,model,num_samples=10000):

    """ Sample the TSIR model without importation and spatial correlation. 
    df is the province dataframe with sia, Z_t, and I_t. model is the dictionary composed during 
    model fitting above. """

    ## Hyper parameters
    n_steps = len(df)
    std_logE = model["std_logE"]

    ## Allocate the appropriate storage
    full_samples = np.zeros((num_samples,n_steps))
    full_samples_S = np.zeros((num_samples,n_steps))
    one_step_samples = np.zeros((num_samples,n_steps))
    one_step_S = np.zeros((num_samples,n_steps))

    ## Set up the initial conditions, accounting for importation
    full_samples[:,0] = df["I_t"].values[0]
    one_step_samples[:,0] = df["I_t"].values[0]
    full_samples_S[:,0] = model["S_bar"] + df["Z_t"].values[0]
    one_step_S[:,0] = model["S_bar"] + df["Z_t"].values[0]

    ## Loop through time
    for i in range(1,n_steps):
        
        ## Time of year for seasonality
        time_in_period = i % model["periodicity"]

        ## Update full projection lambda
        lam = model["t_beta"][time_in_period]*(full_samples_S[:,i-1])*((full_samples[:,i-1])**model["alpha"])

        ## And the one step projection
        lam_one_step = model["t_beta"][time_in_period]*(one_step_S[:,i-1])*((df["I_t"].values[i-1])**model["alpha"])

        ## Sample for new infecteds in both cases
        I_ts = lam*np.exp(std_logE*np.random.normal(size=(num_samples,)))
        S_ts = (full_samples_S[:,i-1]+df["adj_births"].values[i]-I_ts)*(1.-df["sia"].values[i-1])

        ## Update predictions and residuals
        full_samples_S[:,i] = S_ts
        full_samples[:,i] = I_ts
        one_step_samples[:,i] = lam_one_step*np.exp(std_logE*np.random.normal(size=(num_samples,)))
        one_step_S[:,i] = (one_step_S[:,i-1]+df["adj_births"].values[i]-one_step_samples[:,i])*(1.-df["sia"].values[i-1])

        ## Take care of negatives (This happens for large std_logE, which probably shouldn't
        ## be the case? I need a better fix...)
        full_samples[full_samples[:,i] < 0,i] = 0.
        one_step_samples[one_step_samples[:,i] < 0,i] = 0.

    return full_samples, full_samples_S, one_step_samples, one_step_S