#!/usr/bin/env python

import json
import korali
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def get_samples_posterior(filename):

    with open(filename) as f:
        doc = json.load(f)
        samples = np.array(doc["Results"]["Posterior Sample Database"])
    return samples

def gen_normal_samples_from_evaluations(results_path: str,
                                        num_samples_per_x: int=200):

    with open(os.path.join(results_path, 'latest')) as f:
        doc = json.load(f)

    y = np.array([s['Evaluations'] for s in doc['Samples']])
    sigma = np.array([s['sigma'] for s in doc['Samples']])

    num_samples, num_x = y.shape

    ysamples = np.random.normal(loc=y[:,:,np.newaxis], scale=sigma[:,np.newaxis,np.newaxis],
                                size=(num_samples, num_x, num_samples_per_x))

    return ysamples



def generate_samples_xy_model_0(x):
    samples = get_samples_posterior(os.path.join("_korali_results_0", "latest"))

    def model(ks):
        a, sigma = ks["Parameters"]
        ks["Evaluations"] = [a] * len(x)
        ks["sigma"] = sigma

    e = korali.Experiment()
    e['Problem']['Type'] = 'Propagation'
    e['Problem']['Execution Model'] = model

    e['Variables'][0]['Name'] = 'a'
    e['Variables'][0]['Precomputed Values'] = samples[:,0].tolist()
    e['Variables'][1]['Name'] = 'sigma'
    e['Variables'][1]['Precomputed Values'] = samples[:,1].tolist()

    e['Solver']['Type'] = 'Executor'
    e['Solver']['Executions Per Generation'] = 1000

    results_path = '_korali_result_propagation_0'

    e['Console Output']['Verbosity'] = 'Minimal'
    e['File Output']['Path'] = results_path
    e['Store Sample Information'] = True

    k = korali.Engine()
    k.run(e)

    ysamples = gen_normal_samples_from_evaluations(results_path)
    return x, ysamples

def generate_samples_xy_model_1(x):
    samples = get_samples_posterior(os.path.join("_korali_results_1", "latest"))

    def model(ks):
        a, b, sigma = ks["Parameters"]
        ks["Evaluations"] = (a * x + b).tolist()
        ks["sigma"] = sigma

    e = korali.Experiment()
    e['Problem']['Type'] = 'Propagation'
    e['Problem']['Execution Model'] = model

    e['Variables'][0]['Name'] = 'a'
    e['Variables'][0]['Precomputed Values'] = samples[:,0].tolist()
    e['Variables'][1]['Name'] = 'b'
    e['Variables'][1]['Precomputed Values'] = samples[:,1].tolist()
    e['Variables'][2]['Name'] = 'sigma'
    e['Variables'][2]['Precomputed Values'] = samples[:,2].tolist()

    e['Solver']['Type'] = 'Executor'
    e['Solver']['Executions Per Generation'] = 1000

    results_path = '_korali_result_propagation_1'

    e['Console Output']['Verbosity'] = 'Minimal'
    e['File Output']['Path'] = results_path
    e['Store Sample Information'] = True

    k = korali.Engine()
    k.run(e)

    ysamples = gen_normal_samples_from_evaluations(results_path)
    return x, ysamples


def generate_samples_xy_model_2(x):
    samples = get_samples_posterior(os.path.join("_korali_results_2", "latest"))

    def model(ks):
        a, b, c, sigma = ks["Parameters"]
        ks["Evaluations"] = (a * x**2 + b * x + c).tolist()
        ks["sigma"] = sigma

    e = korali.Experiment()
    e['Problem']['Type'] = 'Propagation'
    e['Problem']['Execution Model'] = model

    e['Variables'][0]['Name'] = 'a'
    e['Variables'][0]['Precomputed Values'] = samples[:,0].tolist()
    e['Variables'][1]['Name'] = 'b'
    e['Variables'][1]['Precomputed Values'] = samples[:,1].tolist()
    e['Variables'][2]['Name'] = 'c'
    e['Variables'][2]['Precomputed Values'] = samples[:,2].tolist()
    e['Variables'][3]['Name'] = 'sigma'
    e['Variables'][3]['Precomputed Values'] = samples[:,3].tolist()

    e['Solver']['Type'] = 'Executor'
    e['Solver']['Executions Per Generation'] = 1000

    results_path = '_korali_result_propagation_2'

    e['Console Output']['Verbosity'] = 'Minimal'
    e['File Output']['Path'] = results_path
    e['Store Sample Information'] = True

    k = korali.Engine()
    k.run(e)

    ysamples = gen_normal_samples_from_evaluations(results_path)
    return x, ysamples


def generate_samples_xy_model_3(x):
    samples = get_samples_posterior(os.path.join("_korali_results_3", "latest"))

    def model(ks):
        a, b, c, d, sigma = ks["Parameters"]
        ks["Evaluations"] = (a * x**3 + b * x**2 + c * x + d).tolist()
        ks["sigma"] = sigma

    e = korali.Experiment()
    e['Problem']['Type'] = 'Propagation'
    e['Problem']['Execution Model'] = model

    e['Variables'][0]['Name'] = 'a'
    e['Variables'][0]['Precomputed Values'] = samples[:,0].tolist()
    e['Variables'][1]['Name'] = 'b'
    e['Variables'][1]['Precomputed Values'] = samples[:,1].tolist()
    e['Variables'][2]['Name'] = 'c'
    e['Variables'][2]['Precomputed Values'] = samples[:,2].tolist()
    e['Variables'][3]['Name'] = 'd'
    e['Variables'][3]['Precomputed Values'] = samples[:,3].tolist()
    e['Variables'][4]['Name'] = 'sigma'
    e['Variables'][4]['Precomputed Values'] = samples[:,4].tolist()

    e['Solver']['Type'] = 'Executor'
    e['Solver']['Executions Per Generation'] = 1000

    results_path = '_korali_result_propagation_3'

    e['Console Output']['Verbosity'] = 'Minimal'
    e['File Output']['Path'] = results_path
    e['Store Sample Information'] = True

    k = korali.Engine()
    k.run(e)

    ysamples = gen_normal_samples_from_evaluations(results_path)
    return x, ysamples



if __name__ == '__main__':
    df = pd.read_csv("data.csv")
    xdata = df['x'].to_numpy()
    ydata = df['y'].to_numpy()

    x = np.linspace(-5.2, 5.2, 100)

    generate_samples_functions = [
        generate_samples_xy_model_0,
        generate_samples_xy_model_1,
        generate_samples_xy_model_2,
        generate_samples_xy_model_3
    ]


    for i, generate_samples_xy in enumerate(generate_samples_functions):
        x, ysamples = generate_samples_xy(x)

        fig, ax = plt.subplots()

        for p in [0.99, 0.90, 0.50]:
            y_qlo = np.quantile(ysamples, axis=(0,2), q=0.5-p/2)
            y_qhi = np.quantile(ysamples, axis=(0,2), q=0.5+p/2)
            ax.fill_between(x, y_qlo, y_qhi, label=f"{100*p}% credible intervals")

        y_mean = np.mean(ysamples, axis=(0,2))
        ax.plot(x, y_mean, '--k', label='mean')

        ax.plot(xdata, ydata, 'ok', label='data')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xlim(-5.2, 5.2)
        ax.legend()
        plt.savefig(f"prediction_{i}.pdf", bbox_inches='tight', transparent=True)
        plt.close()
