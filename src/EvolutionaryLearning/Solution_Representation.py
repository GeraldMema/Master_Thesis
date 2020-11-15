import numpy as np


class Solution_Representation:
    """
    Add Description
    """

    def __init__(self, evolutionary_learning_methods, max_no_features, max_no_classifiers):
        self.chromosome = None
        self.max_no_features = max_no_features
        self.max_no_classifiers = max_no_classifiers
        self.representation_method = evolutionary_learning_methods['chromosome_representation']
        if self.representation_method == '1D':
            self.chromosome_length = max_no_features
        elif self.representation_method == '2D':
            self.chromosome_length = max_no_features
        elif self.representation_method == 'dual':
            self.chromosome_length = -1 # TODO
        else:
            self.chromosome_length = -1

    def oneD_representation(self, selected_features):
        """
        In this representation, we present a solution as a 1D numpy array with a following format:
        The first n values are representing the features (n=no_features)

        :param
        selected_features: a list of selected features

        :return:
        chromosome: a numpy array which represent our solution
        """

        self.chromosome = np.zeros(self.max_no_features)
        for feat in selected_features:
            self.chromosome[feat] = 1

    def twoD_representation(self, feat_per_clf):
        """
        In this representation, we present a solution as a 2D numpy array with a following format:
        1) the first dimension (rows) are representing the classifiers
        2) the second dimension (columns) are representing the features

        :param
        feat_per_clf: a list of lists which contains all the selected feature per classifier

        :return:
        chromosome: a 2D numpy array which represent our solution
        """

        self.chromosome = np.zeros((self.max_no_classifiers, self.max_no_features))
        for i in range(self.max_no_classifiers):
            for feat in feat_per_clf[i]:
                self.chromosome[i][feat] = 1

    def dual_representation(self):
        pass
