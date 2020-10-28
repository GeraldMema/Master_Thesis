class Solution_Info:
    """
    Add Description
    """

    def __init__(self, classifiers, features):
        self.classifiers = classifiers
        self.features = features
        self.no_classifiers = len(classifiers)
        self.no_features = len(features)
        self.classifiers_dict = {}
        self.features_dict = {}
        self.set_classifiers()
        self.set_features()

    def set_classifiers(self):
        for idx, clf in enumerate(self.classifiers):
            self.classifiers_dict[idx] = clf

    def set_features(self):
        for idx, feat in enumerate(self.features):
            self.features_dict[idx] = feat
