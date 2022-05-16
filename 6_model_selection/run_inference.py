#!/usr/bin/env python

import korali
import numpy as np
import pandas as pd

def load_data(filename: str):
    df = pd.read_csv(filename)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    return x, y

def make_experiment_0(x, y):
    def model(ks):
        a, sigma = ks["Parameters"]
        ks["Reference Evaluations"] = len(x) * [a]
        ks["Standard Deviation"] = len(x) * [sigma]

    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = y.tolist()
    e["Problem"]["Computational Model"] = model

    e["Variables"]= [{"Name": "a", "Prior Distribution": "Prior a"},
                     {"Name": "sigma", "Prior Distribution": "Prior sigma"}]

    e["Distributions"] = [{"Name": "Prior a",
                           "Type": "Univariate/Uniform",
                           "Minimum": -40.0,
                           "Maximum": +40.0},
                          {"Name": "Prior sigma",
                           "Type": "Univariate/Uniform",
                           "Minimum": 0.0,
                           "Maximum": 40.0}]

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = 5000

    e["File Output"]["Path"] = f"_korali_results_0"
    return e


def make_experiment_1(x, y):
    def model(ks):
        a, b, sigma = ks["Parameters"]
        y_pred = a * x + b
        ks["Reference Evaluations"] = y_pred.tolist()
        ks["Standard Deviation"] = len(y_pred) * [sigma]

    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = y.tolist()
    e["Problem"]["Computational Model"] = model

    e["Variables"]= [{"Name": "a", "Prior Distribution": "Prior a"},
                     {"Name": "b", "Prior Distribution": "Prior b"},
                     {"Name": "sigma", "Prior Distribution": "Prior sigma"}]

    e["Distributions"] = [{"Name": "Prior a",
                           "Type": "Univariate/Uniform",
                           "Minimum": -10.0,
                           "Maximum": +10.0},
                          {"Name": "Prior b",
                           "Type": "Univariate/Uniform",
                           "Minimum": -40.0,
                           "Maximum": +40.0},
                          {"Name": "Prior sigma",
                           "Type": "Univariate/Uniform",
                           "Minimum": 0.0,
                           "Maximum": 40.0}]

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = 5000

    e["File Output"]["Path"] = f"_korali_results_1"
    return e


def make_experiment_2(x, y):
    def model(ks):
        a, b, c, sigma = ks["Parameters"]
        y_pred = a * x**2 + b * x + c
        ks["Reference Evaluations"] = y_pred.tolist()
        ks["Standard Deviation"] = len(y_pred) * [sigma]

    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = y.tolist()
    e["Problem"]["Computational Model"] = model

    e["Variables"]= [{"Name": "a", "Prior Distribution": "Prior a"},
                     {"Name": "b", "Prior Distribution": "Prior b"},
                     {"Name": "c", "Prior Distribution": "Prior c"},
                     {"Name": "sigma", "Prior Distribution": "Prior sigma"}]

    e["Distributions"] = [{"Name": "Prior a",
                           "Type": "Univariate/Uniform",
                           "Minimum": -10.0,
                           "Maximum": +10.0},
                          {"Name": "Prior b",
                           "Type": "Univariate/Uniform",
                           "Minimum": -10.0,
                           "Maximum": +10.0},
                          {"Name": "Prior c",
                           "Type": "Univariate/Uniform",
                           "Minimum": -40.0,
                           "Maximum": +40.0},
                          {"Name": "Prior sigma",
                           "Type": "Univariate/Uniform",
                           "Minimum": 0.0,
                           "Maximum": 40.0}]

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = 5000

    e["File Output"]["Path"] = f"_korali_results_2"
    return e


def make_experiment_3(x, y):
    def model(ks):
        a, b, c, d, sigma = ks["Parameters"]
        y_pred = a * x**3 + b * x**2 + c * x + d
        ks["Reference Evaluations"] = y_pred.tolist()
        ks["Standard Deviation"] = len(y_pred) * [sigma]

    e = korali.Experiment()

    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = y.tolist()
    e["Problem"]["Computational Model"] = model

    e["Variables"]= [{"Name": "a", "Prior Distribution": "Prior a"},
                     {"Name": "b", "Prior Distribution": "Prior b"},
                     {"Name": "c", "Prior Distribution": "Prior c"},
                     {"Name": "d", "Prior Distribution": "Prior d"},
                     {"Name": "sigma", "Prior Distribution": "Prior sigma"}]

    e["Distributions"] = [{"Name": "Prior a",
                           "Type": "Univariate/Uniform",
                           "Minimum": -10.0,
                           "Maximum": +10.0},
                          {"Name": "Prior b",
                           "Type": "Univariate/Uniform",
                           "Minimum": -10.0,
                           "Maximum": +10.0},
                          {"Name": "Prior c",
                           "Type": "Univariate/Uniform",
                           "Minimum": -10.0,
                           "Maximum": +10.0},
                          {"Name": "Prior d",
                           "Type": "Univariate/Uniform",
                           "Minimum": -40.0,
                           "Maximum": +40.0},
                          {"Name": "Prior sigma",
                           "Type": "Univariate/Uniform",
                           "Minimum": 0.0,
                           "Maximum": 40.0}]

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = 5000

    e["File Output"]["Path"] = f"_korali_results_3"
    return e



if __name__ == '__main__':

    x, y = load_data('data.csv')

    experiments = [make_experiment_0(x, y),
                   make_experiment_1(x, y),
                   make_experiment_2(x, y),
                   make_experiment_3(x, y)]

    # Experiments need a Korali Engine object to be executed
    k = korali.Engine()

    k["Conduit"]["Type"] = "Concurrent"
    k["Conduit"]["Concurrent Jobs"] = 4

    # Run the inference
    for e in experiments:
        k.run(e)
