#!/usr/bin/env python

import json
import numpy as np
import os

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
