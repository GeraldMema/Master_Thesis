from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


def possible_classifiers():
    return {
        'DT2': DecisionTreeClassifier(max_depth=2),
        'DT3': DecisionTreeClassifier(max_depth=3),
        'DT4': DecisionTreeClassifier(max_depth=4),
        'DT5': DecisionTreeClassifier(max_depth=5),
        'DT6': DecisionTreeClassifier(max_depth=6),
        'DT7': DecisionTreeClassifier(max_depth=7),
        'LR': LogisticRegression(),
        'SGD': SGDClassifier(),
        'MLP': MLPClassifier(alpha=1, max_iter=1000),
        'SVM': SVC(kernel="linear", C=0.025),
        'GaussianNB': GaussianNB(),
        'Perceptron': Perceptron(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'GPClassifier': GaussianProcessClassifier(1.0 * RBF(1.0)),
        'KNN1': KNeighborsClassifier(1),
        'KNN3': KNeighborsClassifier(3),
        'KNN5': KNeighborsClassifier(5),
    }



class Classifiers:

    def __init__(self, multiple_classification_models_params):
        self.classifier_models = {}
        self.classifier_dict = {}
        self.comparison_classifiers = {}
        self.max_depth = multiple_classification_models_params['max_depth']
        self.all_possible_classifiers = possible_classifiers()
        self.selected_classifiers = list(multiple_classification_models_params['selected_classifiers'])
        self.classifier_selection()

    def set_comparison_classifiers(self, n_estimators):
        self.comparison_classifiers = {
            'XGBoost': XGBClassifier(n_estimators = n_estimators),
            'GBoost': GradientBoostingClassifier(n_estimators = n_estimators, max_depth=self.max_depth, random_state=0),
            'RF': RandomForestClassifier(n_estimators=n_estimators, max_depth=self.max_depth, random_state=0),
            'DT': DecisionTreeClassifier(max_depth=self.max_depth)
        }

    def classifier_selection(self):
        i = 0
        for clf in self.selected_classifiers:
            self.classifier_models[clf] = self.all_possible_classifiers[clf]
            self.classifier_dict[i] = clf
            i = i + 1

