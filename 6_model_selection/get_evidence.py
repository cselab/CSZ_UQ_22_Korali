#!/usr/bin/env python

import json
import numpy as np
import os

def get_get_evidence(filename):
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

    log_evidences = []
    for i in range(4):
        filename = os.path.join(f"_korali_results_{i}", "latest")
        with open(filename) as f:
            doc = json.load(f)
        log_evidence = doc["Results"]["Log Evidence"]
        log_evidences.append(log_evidence)
        print(f"Model {i}: log evidence = {log_evidence}")

    id_best = np.argmax(log_evidences)
    print(f"Model {id_best} describes best the data.")
