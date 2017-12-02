import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


class DataSet(object):

    def __init__(self, X, y, dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        assert X.shape[0] == y.shape[0]
        self._num_examples = X.shape[0]
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0

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
            self._y = self._y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]


def read_data_sets(filename, 
                   label_name, 
                   dtype=dtypes.float32,                   
                   test_frac=0.1,
                   validation_frac=0.1):
    # read and encode data
    df = pd.read_csv(filename, index_col="sample_id")
    X = df[df.columns[3:]]
    X = X.as_matrix()
    y_raw = df[label_name]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # shuffle data
    perm = np.arange(df.shape[0])
    np.random.shuffle(perm)
    X = X[perm]
    y = y[perm]

    # split to train, validation and test
    test_size = int(test_frac * df.shape[0])
    validation_size = int(validation_frac * df.shape[0])
    X_test, y_test = X[:test_size], y[:test_size]
    X_validation = X[test_size:(test_size+validation_size)]
    y_validation = y[test_size:(test_size+validation_size)]
    X_train = X[test_size + validation_size:]
    y_train = y[test_size + validation_size:]

    # cast to dataset class
    train = DataSet(X_train, y_train, dtype=dtype)
    validation = DataSet(X_validation, y_validation, dtype=dtype)
    test = DataSet(X_test, y_test, dtype=dtype)

    return base.Datasets(train=train, validation=validation, test=test)

