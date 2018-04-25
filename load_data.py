import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
import tensorflow as tf


PATH = "/home/molly/Desktop/DeepTCGA/data"


class DataSet(object):
    def __init__(self, df):
        self.df = df
        self.X_cols = [i for i in df.columns if "label" not in i]
        self.X = df[self.X_cols].as_matrix().astype(np.float32)
        self.y = self.extract_labels(df) 
        self.data_dict = self.y.copy()
        self.data_dict.update({"X": self.X})
        self.num_features = self.X.shape[1]
        self.label_classes = {i:j.shape[1] for i, j in self.y.items()}
 
    def extract_labels(self, df):
        """ replace null values with AAAAA which is always encoded as first col"""
        label_dict = {}
        for label_name in ["tissue", "tumor", "gender"]:
            this_label = df["label_{0}".format(label_name)].replace(np.nan, "AAAAA")
            encoded_label = self.ohe_categorical_label(this_label)
            label_dict[label_name] = encoded_label
        
        # MinMax normalize age label, replace nan with -1
        not_null_idx = df[~df["label_age"].isnull()].index
        ages = df.loc[not_null_idx]["label_age"].as_matrix().reshape(-1, 1)
        df.loc[not_null_idx, "label_age"] = self.norm_numerical_label(ages)
        label_dict["age"] = df["label_age"].replace(np.nan, -1).as_matrix().reshape(-1, 1)
        return label_dict
    
    def extract_STL(self, label_name, one_hot=False):
        """ throw away null values for both X and y"""
        if label_name == "age" and one_hot:
            raise Exception("can't one-hot encode age, which is continuous")
        data_dict = {}
        label = "label_" + label_name
        cols = [label] + self.X_cols
        this_df = self.df[cols].dropna()
        X = this_df[self.X_cols].as_matrix().astype(np.float32)
        y_raw = this_df[label].as_matrix()
        if label_name == "age":
            y = self.norm_numerical_label(y_raw.reshape(-1, 1))
        else:
            if one_hot:
                y = self.ohe_categorical_label(y_raw)
            else:
                y = self.encode_categorical_label(y_raw)
        return (X, y)

    def ohe_categorical_label(self, label_array):
        lb = LabelBinarizer()
        encoded_label = lb.fit_transform(label_array)
        if encoded_label.shape[1] == 1:
            # redudantly encode binary labels with two columns for later speediency
            reverse = [1-i for i in encoded_label[:, 0]]
            encoded_label = np.insert(encoded_label, 0, reverse, axis=1)
        return encoded_label

    def encode_categorical_label(self, label_array):
        lb = LabelEncoder()
        encoded_label = lb.fit_transform(label_array)
        return encoded_label
    
    def norm_numerical_label(self, label_array):
        scaler = MinMaxScaler()
        normed_label = scaler.fit_transform(label_array)
        return normed_label



class SplitSet(object):
    def __init__(self):
        self.test = None
        self.cv = []

    def get_first_fold(self):
        """ use 1st fold of train, validation data by default"""
        self.train = self.cv[0][0]
        self.val = self.cv[0][1]

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
    data_path = "{0}/train_val_test_split/seed{1}".format(PATH, random_state)
    
    # read data
    if filename[-3:] == "csv":
        df_X = pd.read_csv(filename, index_col="sample_id")
    elif filename[-3:] == "hdf":
        df_X = pd.read_hdf(filename, "mRNA")
    else:
        raise 
    df_y = pd.read_csv(PATH+"/labels.csv", index_col="sample_id")
    df = df_y.join(df_X)

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
  

