import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


class DataSet(object):

    def __init__(self, df):
        self._num_examples = df.shape[0]
        self._X = df.iloc[:, 3:].as_matrix()
        self._y = self.extract_labels(df) 
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def extract_labels(self, df):
        label_dict = {}
        for label_name in ["tissue", "tumor", "gender"]:
            lb = LabelBinarizer()
            label_dict[label_name] = lb.fit_transform(df["label_{0}".format(label_name)])

        # redudant encoding gender label with a column 0 and a column 1 for later speediency
        reverse_gender = [1-i for i in label_dict["gender"][:, 0]]
        label_dict["gender"] = np.insert(label_dict["gender"], 0, reverse_gender, axis=1)
        return label_dict


    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y
  
    @property
    def num_examples(self):
        return self._num_examples
  
    @property
    def epochs_completed(self):
        return self._epochs_completed
  
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
        
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._X = self._X[perm]
            for key, value in self._y.items():
                self._y[key] = self._y[key][perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        batch_x = self._X[start:end]
        batch_y = {"tissue": self._y["tissue"][start:end],
                   "gender": self._y["gender"][start:end],
                   "tumor": self._y["tumor"][start:end]}
        return batch_x, batch_y


    def reset_epoch(self):
        self._epochs_completed = 0


def read_data_sets(filename, test_frac=0.1):
    # read and encode data
    df_X = pd.read_csv(filename, index_col="sample_id")
    df_y = pd.read_csv("./data/labels.csv", index_col="sample_id")
    df = df_y.join(df_X).dropna()

    # shuffle data
    perm = np.arange(df.shape[0])
    np.random.shuffle(perm)
    df = df.iloc[perm, :]

    # split to train, validddation and test
    test_size = int(test_frac * df.shape[0])
    df_test = df.iloc[:test_size, :]
    df_validation = df.iloc[test_size:2*test_size, :]
    df_train = df.iloc[2*test_size:, :]
    
    # cast to dataset class
    train = DataSet(df_train)
    validation = DataSet(df_validation)
    test = DataSet(df_test)

    return base.Datasets(train=train, validation=validation, test=test)
