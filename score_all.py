import os
import sys

import numpy as np
import pandas as pd
from sklearn.externals import joblib


def load_model(path):
    return joblib.load(path)


def score(values, thr):
    models_root = '/Users/srikanthreddy/Downloads/test/data/logs/2019-06-02-18-22-14'
    models_paths = [os.path.join(models_root, m) for m in os.listdir(models_root) if m.endswith('.bin')]
    models_paths.sort()

    scores = np.zeros((len(values), ))
    for path in models_paths:
        model = load_model(path)
        scores += model.predict_proba(values)[:, 1]

    scores = scores / len(models_paths)
    labels = np.where(scores > thr, 1, 0)

    return (scores, labels)


def read_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    return (values, column_names)


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'psv'):
            files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Iterate over files.
    for f in files:
        # Load data.
        input_file = os.path.join(input_directory, f)
        (values, column_names) = read_data(input_file)

        # generate predictions
        thr = 0.29
        example = pd.DataFrame(values, columns=None, copy=True)

        example.ffill(inplace=True)
        example.bfill(inplace=True)
        example.fillna(0, inplace=True)

        (scores, labels) = score(example.values, thr)

        # Save results.
        output_file = os.path.join(output_directory, f)
        with open(output_file, 'w') as f:
            f.write('PredictedProbability|PredictedLabel\n')
            for (s, l) in zip(scores, labels):
                f.write('%g|%d\n' % (s, l))
