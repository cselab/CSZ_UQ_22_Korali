#!/usr/bin/env python

import numpy as np
import pandas as pd

x = np.linspace(-5, 5, 20)
y = np.random.normal(x**2 - 3 * x + 2, 2)

df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('data.csv', index=False)
