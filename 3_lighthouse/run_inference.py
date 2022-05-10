#!/usr/bin/env python

import korali
import numpy as np
import pandas as pd

def load_data(filename: str):
    df = pd.read_csv(filename)
    x = df['x'].to_numpy()
    return x

if __name__ == '__main__':

    e = korali.Experiment()

    x = load_data('data.csv')

    # Custom likelihood: must compute the likelihood of the data
    # given the parameters
    def log_likelihood(ksample):
        a, b = ksample["Parameters"]
        L = np.log(b) * len(x) - np.sum(np.log(b**2 + (x-a)**2))
        ksample["logLikelihood"] = L

    e["Problem"]["Type"] = "Bayesian/Custom"
    e["Problem"]["Likelihood Model"] = log_likelihood

    e["Variables"][0]["Name"] = "a"
    e["Variables"][0]["Prior Distribution"] = "Prior a"

    e["Variables"][1]["Name"] = "b"
    e["Variables"][1]["Prior Distribution"] = "Prior b"

    e["Distributions"][0]["Name"] = "Prior a"
    e["Distributions"][0]["Type"] = "Univariate/Uniform"
    e["Distributions"][0]["Minimum"] = 0
    e["Distributions"][0]["Maximum"] = 50

    e["Distributions"][1]["Name"] = "Prior b"
    e["Distributions"][1]["Type"] = "Univariate/Uniform"
    e["Distributions"][1]["Minimum"] = 10
    e["Distributions"][1]["Maximum"] = 100

    # Configuring TMCMC parameters
    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = 5000

    e["Console Output"]["Verbosity"] = "Detailed"

    # Experiments need a Korali Engine object to be executed
    k = korali.Engine()

    # Run the optimization
    k.run(e)
