#!/usr/bin/env python

import numpy as np
import pandas as pd

x = np.linspace(-1, 1, 10)
y = np.random.normal(3 * x - 2, 0.5)

df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('data.csv', index=False)
