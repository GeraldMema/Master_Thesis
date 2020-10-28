import numpy as np


class Solution_Representation:
    """
    Add Description
    """

    def __init__(self, solution_info, representation_method):
        self.solution_info = solution_info
        self.chromosome = None
        if representation_method == 0:
            self.representation_method = '1D'
        elif representation_method == 1:
            self.representation_method = '2D'
        elif representation_method == 2:
            self.representation_method = 'Dual'  # Paper: Novel Genetic Algorithm with Dual Chromosome Representation
            # for Resource Allocation in Container-Based Clouds
        else:
            self.representation_method = None

    def oneD_representation(self, selected_classifiers, selected_features):
        """
        In this representation, we present a solution as a 1D numpy array with a following format:
        1) the first n values are representing the features (n=no_features)
        2) the last k values are representing the classifiers (k=no_classifiers)
        3) size of list = n+k

        :param
        selected_classifiers: a list of selected classifiers
        selected_features: a list of selected features

        :return:
        chromosome: a numpy array which represent our solution
        """
        max_no_classifiers = self.solution_info.no_classifiers
        max_no_features = self.solution_info.no_features
        self.chromosome = np.zeros(max_no_features+max_no_classifiers)
        for feat in selected_features:
            self.chromosome[feat] = 1
        for clf in selected_classifiers:
            self.chromosome[clf + max_no_features] = 1

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
        max_no_classifiers = self.solution.no_classifiers
        max_no_features = self.solution.no_features
        self.chromosome = np.zeros((max_no_classifiers,max_no_features))

        for i in range(len(feat_per_clf)):
            for feat in feat_per_clf[i]:
                self.chromosome[i][feat] = 1

    def dual_representation(self):
        pass
