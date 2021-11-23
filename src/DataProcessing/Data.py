import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from sklearn.preprocessing import LabelEncoder
from pmlb import fetch_data, classification_dataset_names

import numpy as np
import os

from src.DataProcessing.Data_Preprocessing import Data_Preprocessing
from src.DataProcessing.Data_Transformation import Data_Transformation


class Data:
    """
    Add Description
    """

    def __init__(self, data_params, run, sep=','):
        self.features_dict = {}
        self.dataset = data_params['dataset']
        self.no_trans = data_params['no_transform']
        self.dataset_filename = data_params['path'] + data_params['filename']
        self.sep = sep
        self.ID = None
        self.X_train = None
        self.y_train = None
        self.X_cv = None
        self.y_cv = None
        self.X_test = None
        self.y_test = None
        self.features = None
        self.transformation_method = data_params['normalization']
        self.run_all_datasets = data_params['run_all_datasets']
        self.random_state = run
        self.is_imbalanced = data_params['imbalanced']
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

    def process(self, classification_dataset):
        """
        Add description here
        """
        if not self.run_all_datasets:
            global data
            if self.dataset == 'iris':
                data, target = load_iris(return_X_y=True, as_frame=True)
            elif self.dataset == 'digits':
                data, target = load_digits(return_X_y=True, as_frame=True)
                if self.is_imbalanced:
                    samples_to_delete = list(target[target == 0].sample(n=int(0.75 * len(target[target == 0]))).index)
                    samples_to_delete.extend(
                        list(target[target == 1].sample(n=int(0.75 * len(target[target == 0]))).index))
                    samples_to_delete.extend(
                        list(target[target == 2].sample(n=int(0.75 * len(target[target == 0]))).index))
                    samples_to_delete.extend(
                        list(target[target == 3].sample(n=int(0.75 * len(target[target == 0]))).index))
                    data = data.drop(samples_to_delete)
                    target = target.drop(samples_to_delete)
            elif self.dataset == 'analcatdata_cyyoung9302':
                df = fetch_data(self.dataset)
                target = df.target
                data = df.drop('target', axis=1)
            elif self.dataset == 'analcatdata_cyyoung8092':
                df = fetch_data(self.dataset)
                target = df.target
                data = df.drop('target', axis=1)
            elif self.dataset == 'car_evaluation':

                df = pd.read_csv(self.ROOT_DIR + '/../../' + self.dataset_filename, sep=',',
                                 names=('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target'))
                le = preprocessing.LabelEncoder()
                df['target'] = le.fit_transform(df.target.values)

                target = df.target
                data = df.drop('target', axis=1)
                data = pd.get_dummies(data)

            elif self.dataset == 'backache':
                df = fetch_data(self.dataset)
                target = df.target
                data = df.drop('target', axis=1)
            elif self.dataset == 'soccer':
                df = pd.read_csv('../../../../' + self.dataset_filename)
                df = df[:1000]
                target = df.label
                data = df.drop('label', axis=1)

            elif self.dataset == 'digits_csv':
                df = pd.read_csv('../../../../' + self.dataset_filename)
                df = df[:3000]
                target = df.label
                data = df.drop('label', axis=1)

            elif self.dataset == 'adult':
                df = pd.read_fwf(os.path.dirname(__file__) + '/../../' + self.dataset_filename, sep=', ',
                                 names=["col1", "col2"])
                df[["col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10", "col11", "col12", "col13",
                    "col14", "col15"]] = df.col2.str.split(', ', expand=True)
                df = df[df['col15'].notna()]
                df.col1 = df.col1.str[:-1]
                numeric_cols = ['col1', 'col3', 'col5', 'col11', 'col12', 'col13']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                categ_cols = ['col2', 'col4', 'col6', 'col7', 'col8', 'col9', 'col10', 'col14']
                for col in categ_cols:
                    df[col] = df[col].astype('category')
                # creating instance of labelencoder
                df.col15[df.col15 == '>50'] = ">50K"
                df.col15[df.col15 == '>'] = ">50K"
                df.col15[df.col15 == '>5'] = ">50K"
                df.col15[df.col15 == '<=50'] = "<=50K"
                df.col15[df.col15 == '<='] = "<=50K"
                df.col15[df.col15 == '<=5'] = "<=50K"
                df.col15[df.col15 == '<'] = "<=50K"
                labelencoder = LabelEncoder()
                df['col15'] = labelencoder.fit_transform(df['col15'])
                df = pd.get_dummies(df)
                df = df[:3000]
                target = df.col15
                data = df.drop('col15', axis=1)
            elif self.dataset == 'breast_cancer':
                data, target = load_breast_cancer(return_X_y=True, as_frame=True)
                if self.is_imbalanced:
                    samples_to_delete = list(target[target == 0].sample(n=int(0.75 * len(target[target == 0]))).index)
                    data = data.drop(samples_to_delete)
                    target = target.drop(samples_to_delete)
            elif self.dataset == 'wine':
                data, target = load_wine(return_X_y=True, as_frame=True)
            elif self.dataset == 'other':
                self.get_data(self.data_path)
            else:
                print('Wrong dataset input')
                return

        else:
            d = fetch_data(classification_dataset)
            d.drop_duplicates(inplace=True)
            if len(d)>20000:
                frac = 20000/len(d)
                d = d.sample(frac=frac)
            # if (len(d.columns) > 10) & (len(d.columns) < 30):
            # d = d[:5000]
            target = d.target
            data = d.drop(['target'], axis=1)

        # data = data[:20000]

        # Transform data
        if self.no_trans:
            transformed_data = data
        else:
            transformer = Data_Transformation(self.transformation_method)
            transformed_data = transformer.transform_data(data)

        # WARNING: TEST has more data than CV
        # split test CV
        X_train, self.X_test, y_train, self.y_test = train_test_split(transformed_data, target, test_size=0.1,
                                                                      random_state=self.random_state)
        return X_train, y_train

    def split_data(self, random_state, X_train, y_train):

        self.X_train, self.X_cv, self.y_train, self.y_cv = train_test_split(X_train, y_train, test_size=0.1,
                                                                            random_state=random_state)

        self.features = self.X_train.columns
        for i in range(len(self.features)):
            self.features_dict[i] = self.features[i]

    def get_data(self, path):
        os.chdir('../../../../')
        df = pd.read_csv(path, sep=',')
        dp = Data_Preprocessing(df)
        self.X_train = dp.X_train
        self.y_train = dp.y_train
        self.X_cv = dp.X_cv
        self.y_cv = dp.y_cv
        self.X_test_ = dp.X_test_
        self.y_test = dp.y_test
        self.features_dict = dp.features_dict

        return dp.data, dp.target
