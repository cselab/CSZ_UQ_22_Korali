#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("data.csv")
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(x, y, 'ok')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.savefig("raw_data.pdf", bbox_inches='tight', transparent=True)
