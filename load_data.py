import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf



class DataSet(object):
    def __init__(self, df):
        self.X = df.iloc[:, 3:].as_matrix().astype(np.float32)
        self.y = self.extract_labels(df) 
        self.data_dict = self.y.copy()
        self.data_dict.update({"X": self.X})
        self.num_features = self.X.shape[1]
        self.label_classes = {i:j.shape[1] for i, j in self.y.items()}
 
    def extract_labels(self, df):
        label_dict = {}
        for label_name in ["tissue", "tumor", "gender"]:
            lb = LabelBinarizer()
            label_dict[label_name] = lb.fit_transform(df["label_{0}".format(label_name)])
            if label_dict[label_name].shape[1] == 1:
            # redudantly encode binary labels with two columns for later speediency
                reverse = [1-i for i in label_dict[label_name][:, 0]]
                label_dict[label_name] = np.insert(label_dict[label_name], 0, reverse, axis=1)
        return label_dict


class SplitSet(object):
    def __init__(self):
        self.test = None
        self.cv = []

    def get_first_fold(self):
        """ use 1st fold of train, validation data by default"""
        self.train = self.cv[0][0]
        self.val = self.cv[0][0]

    def prep_test_batch(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data.data_dict)
        dataset = dataset.repeat(-1)
        dataset = dataset.batch(data.X.shape[0])
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        return next_batch, iterator

    def prep_batch(self, fold=0, batch_size=128, count_by="step"):
        """ allow for counting epoch number by catching exception"""
        train_data, val_data = self.cv[fold]
        val_all, val_iter = self.prep_test_batch(val_data)
        train_all, train_iter_all = self.prep_test_batch(train_data)
        dataset = tf.data.Dataset.from_tensor_slices(train_data.data_dict)
        dataset = dataset.shuffle(buffer_size=10000)
        if count_by == "step":
            dataset = dataset.repeat(-1)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2*batch_size) # does this do anything?
        train_iter = dataset.make_initializable_iterator()
        train_next_batch = train_iter.get_next()
        return (train_next_batch, train_iter, 
                val_all, val_iter,
                train_all, train_iter_all)


def read_data_sets(filename, random_state=0):
    data = SplitSet() 
    data_path = "./data/train_val_test_split/seed{0}".format(random_state)
    
    # read data
    if filename[-3:] == "csv":
        df_X = pd.read_csv(filename, index_col="sample_id")
    elif filename[-3:] == "hdf":
        df_X = pd.read_hdf(filename)
    else:
        raise 
    df_y = pd.read_csv("./data/labels.csv", index_col="sample_id")
    df = df_y.join(df_X).dropna()

    # save all data into SplitSet object
    data.all = DataSet(df)

    # seperate to train and val from test
    test_file = os.path.join(data_path, "test.csv")
    df_test = df.loc[get_sample_ids(test_file)]
    data.test = DataSet(df_test) 

    # seperate train from val in cross validation
    cv_samples = {}
    for fold in range(5):
        train_file = os.path.join(data_path, "train_fold{0}.csv".format(fold))
        val_file = os.path.join(data_path, "val_fold{0}.csv".format(fold))
        df_train = df.loc[get_sample_ids(train_file)]
        df_val = df.loc[get_sample_ids(val_file)]
        data.cv.append((DataSet(df_train), DataSet(df_val)))
   
    data.get_first_fold() # load fold 1 of cv as default train, validation 
    return data


def get_sample_ids(filename):
    with open(filename, "r") as f:
        return f.read().strip("\n").split("\n")
  

