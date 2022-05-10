#!/usr/bin/env python

import numpy as np
import pandas as pd

nsamples=500

a = 20
b = 50

theta = np.random.uniform(-np.pi/2, np.pi/2, nsamples)
x = a + b * np.tan(theta)

df = pd.DataFrame({'x': x})
df.to_csv('data.csv', index=False)
