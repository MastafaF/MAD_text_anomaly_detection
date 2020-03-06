# Mastafa
#!/usr/bin/python3
import pandas as pd
import numpy as np

import pickle

import torch
from torch.utils.data import Dataset
import random
import os
import sys


# get environment
# export MAD='/Users/foufamastafa/Documents/master_thesis_KTH/MAD_anomaly_detection'
assert os.environ.get('MAD'), 'Please set the environment variable MAD'
MAD = os.environ['MAD']
DATA_PATH = MAD + "/data/"
sys.path.append(MAD + '/data/')

random.seed(1995)


class AnomalyDataset(Dataset):
    def __init__(self, file_relative_path, train_size = 0.5, test_size = 0.5):
        # Data loading
        # We expect a tsv file here with columns : txt labels
        df = pd.read_csv(DATA_PATH + file_relative_path) # The tsv file should have an index that has been reset
        df = df.reset_index(drop = True)
        X = df.loc[:, 'txt']
        Y = df.loc[:, 'labels']
        Y = Y.values
        self.df = df
        self.X = X
        self.Y = Y
        self.nb_samples = df.shape[0]
        # We can go for the opposite as well but since in MAD reference class is Normal Class
        # We choose to go for the setting where label(normal, normal) = label(normal) = 1 (similar)
        # label(anomaly, normal) = label(anomaly) = 0 (dissimilar)
        self.int2label = {"normal": 1, "anomaly": 0}
        self.train_size = train_size
        self.test_size = test_size
        self.split = [0.5, 0.5] # [train_ratio, test_ratio]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.nb_samples

    def get_train_test_data(self, split_arr):
        """
        :return: train_df, test_df, split_idx dictionary ({'train': idx_train, 'test':idx_test})
        """

        def _split_indices(data, split):
            storage = {'train': [], 'test': []}
            data_size = len(data)
            train_size = round(data_size * split[0])
            examples = range(len(data))
            storage['train'] = random.sample(examples, train_size)
            storage['test'] = [ex for ex in examples if ex not in storage['train']]
            return storage

        def split_data(dataframe, split):
            """
            Split the data into a training set
            and a test set according to 'train_size'
            Args:
                dataframe: (pandas.Dataframe)
                split: (list of float) train/valid/test split
            """
            split_idx = _split_indices(dataframe, split)
            train_data = dataframe.iloc[split_idx['train']]
            test_data = dataframe.iloc[split_idx['test']]
            return train_data, test_data, split_idx


        train_data, test_data, split_idx_dic = split_data(self.df, split = split_arr)
        return train_data, test_data, split_idx_dic
