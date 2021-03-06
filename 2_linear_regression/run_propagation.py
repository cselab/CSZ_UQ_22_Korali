#!/usr/bin/env python

import json
import korali
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def get_samples_posterior(filename):
    a = []
    b = []
    sigma = []
    with open(filename) as f:
        doc = json.load(f)

        for sample in doc["Results"]["Posterior Sample Database"]:
            a.append(sample[0])
            b.append(sample[1])
            sigma.append(sample[2])
    return a, b, sigma

def generate_samples_xy():
    a, b, sigma = get_samples_posterior(os.path.join("_korali_result", "latest"))
    num_x = 100
    x = np.linspace(-1.2, 1.2, num_x)

    def model(ks):
        a, b, sigma = ks["Parameters"]
        ks["Evaluations"] = (a * x + b).tolist()

        # korali samples can store arbitrary data in addition to the required fields.
        # here we store the sampled sigma for each sample.
        ks["sigma"] = sigma

    e = korali.Experiment()
    e['Problem']['Type'] = 'Propagation'
    e['Problem']['Execution Model'] = model

    e['Variables'][0]['Name'] = 'a'
    e['Variables'][0]['Precomputed Values'] = a
    e['Variables'][1]['Name'] = 'b'
    e['Variables'][1]['Precomputed Values'] = b
    e['Variables'][2]['Name'] = 'sigma'
    e['Variables'][2]['Precomputed Values'] = sigma

    e['Solver']['Type'] = 'Executor'
    e['Solver']['Executions Per Generation'] = 1000

    results_path = '_korali_result_propagation'

    e['Console Output']['Verbosity'] = 'Minimal'
    e['File Output']['Path'] = results_path
    e['Store Sample Information'] = True

    k = korali.Engine()
    k.run(e)

    # read the samples from the produced file
    with open(os.path.join(results_path, 'latest')) as f:
        doc = json.load(f)

    y = np.array([s['Evaluations'] for s in doc['Samples']])
    sigma = np.array([s['sigma'] for s in doc['Samples']])

    num_samples_per_x = 200

    ysamples = np.random.normal(loc=y[:,:,np.newaxis], scale=sigma[:,np.newaxis,np.newaxis],
                                size=(len(a), num_x, num_samples_per_x))

    return x, ysamples



if __name__ == '__main__':
    x, ysamples = generate_samples_xy()

    fig, ax = plt.subplots()


    for p in [0.99, 0.90, 0.50]:
        y_qlo = np.quantile(ysamples, axis=(0,2), q=0.5-p/2)
        y_qhi = np.quantile(ysamples, axis=(0,2), q=0.5+p/2)
        ax.fill_between(x, y_qlo, y_qhi, label=f"{100*p}% credible intervals")

    y_mean = np.mean(ysamples, axis=(0,2))
    ax.plot(x, y_mean, '--k', label='mean')

    df = pd.read_csv("data.csv")
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    ax.plot(x, y, 'ok', label='data')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-1.2, 1.2)
    ax.legend(frameon=False)
    plt.savefig("prediction.pdf", bbox_inches='tight', transparent=True)
