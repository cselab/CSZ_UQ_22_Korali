#!/usr/bin/env python

import korali
import numpy as np
import pandas as pd

def load_data(filename: str):
    df = pd.read_csv(filename)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    return x, y

if __name__ == '__main__':

    e = korali.Experiment()

    x, y = load_data('data.csv')

    def model(ksample):
        a, b, sigma = ksample["Parameters"]
        y_pred = a * x + b
        ksample["Reference Evaluations"] = y_pred.tolist()
        ksample["Standard Deviation"] = len(y_pred) * [sigma]

    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = y.tolist()
    e["Problem"]["Computational Model"] = model

    # Configure the variables and their prior distributions

    e["Variables"][0]["Name"] = "a"
    e["Variables"][0]["Prior Distribution"] = "Prior a"

    e["Variables"][1]["Name"] = "b"
    e["Variables"][1]["Prior Distribution"] = "Prior b"

    e["Variables"][2]["Name"] = "sigma"
    e["Variables"][2]["Prior Distribution"] = "Prior sigma"

    e["Distributions"][0]["Name"] = "Prior a"
    e["Distributions"][0]["Type"] = "Univariate/Uniform"
    e["Distributions"][0]["Minimum"] = 1.0
    e["Distributions"][0]["Maximum"] = 5.0

    e["Distributions"][1]["Name"] = "Prior b"
    e["Distributions"][1]["Type"] = "Univariate/Uniform"
    e["Distributions"][1]["Minimum"] = -5.0
    e["Distributions"][1]["Maximum"] = +1.0

    e["Distributions"][2]["Name"] = "Prior sigma"
    e["Distributions"][2]["Type"] = "Univariate/Uniform"
    e["Distributions"][2]["Minimum"] = 0.0
    e["Distributions"][2]["Maximum"] = 2.0

    # Configuring TMCMC parameters
    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = 5000

    # Experiments need a Korali Engine object to be executed
    k = korali.Engine()

    # Setup concurrent conduit
    k["Conduit"]["Type"] = "Concurrent"
    k["Conduit"]["Concurrent Jobs"] = 4

    # Run the optimization
    k.run(e)
