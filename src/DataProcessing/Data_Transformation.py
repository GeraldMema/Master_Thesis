from sklearn import preprocessing
import pandas as pd


class Data_Transformation:
    def __init__(self, transformation_method=None):
        self.transformation_method = transformation_method
        self.transformed_data = None

    def transform_data(self, data):
        if self.transformation_method == "MinMax":
            scale = preprocessing.MinMaxScaler()
        elif self.transformation_method == "Standardization":
            scale = preprocessing.StandardScaler()
        elif self.transformation_method == "Normalization":
            scale = preprocessing.Normalizer()
        else:
            scale = None
        if scale:
            scaled_data = scale.fit_transform(data.values)
            return pd.DataFrame(scaled_data, columns=data.columns)
