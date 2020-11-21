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


def possible_classifiers(random_state):
    return {
        'DT2': DecisionTreeClassifier(max_depth=2, random_state=random_state),
        'DT3': DecisionTreeClassifier(max_depth=3, random_state=random_state),
        'DT4': DecisionTreeClassifier(max_depth=4, random_state=random_state),
        'DT5': DecisionTreeClassifier(max_depth=5, random_state=random_state),
        'DT6': DecisionTreeClassifier(max_depth=6, random_state=random_state),
        'DT7': DecisionTreeClassifier(max_depth=7, random_state=random_state),
        'LR': LogisticRegression(random_state=random_state),
        'SGD': SGDClassifier(random_state=random_state),
        'MLP': MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
        'SVM': SVC(kernel="linear", C=0.025, random_state=random_state),
        'GaussianNB': GaussianNB(),
        'Perceptron': Perceptron(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'GPClassifier': GaussianProcessClassifier(1.0 * RBF(1.0)),
        'KNN1': KNeighborsClassifier(1),
        'KNN3': KNeighborsClassifier(3),
        'KNN5': KNeighborsClassifier(5),
    }


class Classifiers:

    def __init__(self, multiple_classification_models_params, evolutionary_learning_params):
        self.classifier_models = {}
        self.classifier_dict = {}
        self.comparison_classifiers = {}
        self.no_classifiers = evolutionary_learning_params['population_size']
        self.max_depth = multiple_classification_models_params['max_depth']
        self.selected_classifier = list(multiple_classification_models_params['selected_classifiers'])[0]
        # select classifiers with the generation random state
        self.classifier_selection()

    def set_comparison_classifiers(self, n_estimators, random_state):
        self.comparison_classifiers = {
            'XGBoost': XGBClassifier(n_estimators=n_estimators, max_depth=self.max_depth, random_state=random_state),
            'GBoost': GradientBoostingClassifier(n_estimators=n_estimators, max_depth=self.max_depth, random_state=random_state),
            'RF': RandomForestClassifier(n_estimators=n_estimators, max_depth=self.max_depth, random_state=random_state),
            'DT': DecisionTreeClassifier(max_depth=self.max_depth, random_state=random_state)
        }

    def classifier_selection(self):
        clf = self.selected_classifier
        for i in range(self.no_classifiers):
            self.classifier_models[clf + str(i + 1)] = possible_classifiers(0)[clf]  # TODO --> ASK KOSTAS ABOUT RANDOM STATE
            self.classifier_dict[i] = clf + str(i + 1)
