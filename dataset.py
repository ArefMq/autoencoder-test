import pandas as pd
from os import path
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split

col_names = [
    'Number of times pregnant',
    'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
    'Diastolic blood pressure (mm Hg)',
    'Triceps skin fold thickness (mm)',
    '2-Hour serum insulin (mu U/ml)',
    'Body mass index (weight in kg/(height in m)^2)',
    'Diabetes pedigree function',
    'Age (years)',
    'Class variable',
]

FEATURE_LIST = col_names[:-1]


def load_regression_dataset(test_size=0.2, feature_set=None):
    data_set = pd.read_csv(path.join(".", "pima-indians-diabetes.csv"), header=None, names=col_names)

    if feature_set is None:
        feature_set = FEATURE_LIST

    x = data_set[feature_set]  # Features
    y = data_set['Class variable']  # Target variable
    return train_test_split(x, y, test_size=test_size)


def load_normalized_dataset(feature_set=None):
    x_train, x_test, y_train, y_test = load_regression_dataset(feature_set=feature_set)
    return np.array([normalize(n) for n in x_train.values.transpose()]).transpose(), \
           np.array([normalize(n) for n in x_test.values.transpose()]).transpose(), \
           y_train, y_test


def normalize(array):
    v_min = np.min(array)
    v_max = np.max(array)
    return (array - v_min) / (v_max - v_min)
