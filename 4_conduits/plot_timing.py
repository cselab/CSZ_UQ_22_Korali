#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    parser.add_argument("out", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    n = df["n"]
    time = df["time"]
    speedup = time[0] / time

    fig, ax = plt.subplots()
    ax.plot(n, speedup, '-ok', label='korali')
    ax.plot(n, n, '--k', label='ideal')
    ax.set_xlabel("number of cores")
    ax.set_ylabel("speedup")
    ax.legend(frameon=False)
    ax.set_xlim(1,4)
    ax.set_ylim(1,4)
    ax.set_xticks(n)
    plt.savefig(args.out, transparent=True)
