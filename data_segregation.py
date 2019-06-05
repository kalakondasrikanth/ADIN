import pandas as pd
import numpy as np
import pickle
import tqdm
import sklearn.preprocessing
import os
from path.path_utils import project_root

def segregation(training_files):

    lengths = []
    is_sepsis = []
    all_data = np.zeros((1552210, 42))
    ind = 0
    training_examples = []
    for i, training_file in enumerate(tqdm.tqdm(training_files)):
        example = pd.read_csv(training_file, sep=',')
        example['seg_id'] = i
        training_examples.append(example)
        is_sepsis.append(1 if 1 in example['SepsisLabel'].values else 0)

        lengths.append(len(example))

        all_data[ind:ind+len(example), :] = example.values
        ind += len(example)
    all_data = pd.DataFrame(all_data, columns=example.columns.values, index=None)

    all_data.to_csv(os.path.join(project_root(), 'data', 'processed', 'training_concatenated.csv'), index=False)
    ss = sklearn.preprocessing.StandardScaler()
    all_data = pd.DataFrame(ss.fit_transform(all_data), columns=all_data.columns.values)

    with open(os.path.join(project_root(), 'data', 'processed', 'lengths.txt'), 'w') as f:
        [f.write('{}\n'.format(l)) for l in lengths]
    with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis.txt'), 'w') as f:
        [f.write('{}\n'.format(l)) for l in is_sepsis]

    with open(os.path.join(project_root(), 'data', 'processed', 'training_raw.pickle'), 'wb') as f:
        pickle.dump(training_examples, f)

    training_examples = []
    for training_file in tqdm.tqdm(training_files):
        example = pd.read_csv(training_file, sep=',')
        example.ffill(inplace=True)
        example.bfill(inplace=True)
        example.fillna(0, inplace=True)
        training_examples.append(example)

    with open(os.path.join(project_root(), 'data', 'processed', 'training_filled.pickle'), 'wb') as f:
        pickle.dump(training_examples, f)


if __name__ == '__main__':

    data_path = os.path.join(project_root(), 'data', 'processed', 'training')
    training_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    training_files.sort()
    segregation(training_files)
