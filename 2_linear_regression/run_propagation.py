#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import norm

def get_korali_samples(filename):
    a = []
    b = []
    sigma = []

    with open(filename) as f:
        doc = json.load(f)

        for sample in doc["Results"]["Posterior Sample Database"]:
            a.append(sample[0])
            b.append(sample[1])
            sigma.append(sample[2])

    return np.array(a), np.array(b), np.array(sigma)

if __name__ == '__main__':

    df = pd.read_csv("data.csv")
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()

    a, b, sigma = get_korali_samples(os.path.join("_korali_result", "latest"))

    numx = 200
    numy = 500

    x_ = np.linspace(np.min(x), np.max(x), numx)
    y_ = np.linspace(-7, 3, numy)

    y_cdf = np.zeros((numx,numy))

    for a_, b_, sigma_ in zip(a, b, sigma):
        y_cdf += norm.cdf(y_[np.newaxis,:], loc=(a_*x_+b_)[:,np.newaxis], scale=sigma_)
    y_cdf /= len(a)

    y_median = y_[np.argmin(np.abs(y_cdf-0.5), axis=1).flatten()]
    y_q05 = y_[np.argmin(np.abs(y_cdf-0.05), axis=1).flatten()]
    y_q95 = y_[np.argmin(np.abs(y_cdf-0.95), axis=1).flatten()]

    fig, ax = plt.subplots()
    ax.fill_between(x_, y_q05, y_q95, alpha=0.2)
    ax.plot(x_, y_median)
    ax.plot(x, y, '+k')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-1,1)
    plt.savefig("prediction.pdf", bbox_inches='tight', transparent=True)
