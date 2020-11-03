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
        'DT3': DecisionTreeClassifier(max_depth=3),
        'DT4': DecisionTreeClassifier(max_depth=4),
        'DT5': DecisionTreeClassifier(max_depth=5),
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


def comparison_classifiers():
    return {
        'XGBoost': XGBClassifier(),
        'GBoost': GradientBoostingClassifier(max_depth=5, random_state=0),
        'RF': RandomForestClassifier(max_depth=5, random_state=0),
        'DT': DecisionTreeClassifier(max_depth=5)
    }


class Classifiers:

    def __init__(self, multiple_classification_models_params):
        self.classifier_models = {}
        self.classifier_dict = {}
        self.all_possible_classifiers = possible_classifiers()
        self.comparison_classifiers = comparison_classifiers()
        self.selected_classifiers = list(multiple_classification_models_params['selected_classifiers'])
        self.classifier_selection()

    def classifier_selection(self):
        i = 0
        for clf in self.selected_classifiers:
            self.classifier_models[clf] = self.all_possible_classifiers[clf]
            self.classifier_dict[i] = clf
            i = i + 1

