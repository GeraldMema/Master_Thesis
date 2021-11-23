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



class Classifiers:

    def __init__(self, multiple_classification_models_params, evolutionary_learning_params,width):
        self.classifier_models = {}
        self.classifier_dict = {}
        self.comparison_classifiers = {}
        self.no_classifiers = evolutionary_learning_params['population_size']
        self.max_depth = multiple_classification_models_params['max_depth']
        self.min_sample = multiple_classification_models_params['min_sample']
        self.selected_classifier = list(multiple_classification_models_params['selected_classifiers'])[0]
        # select classifiers with the generation random state
        self.main_clf = None
        self.classifier_selection(width)

    def possible_classifiers(self, random_state, name):
        # random_state=42
        return {
            name: DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_sample, random_state=random_state),
            'LR': LogisticRegression(random_state=random_state),
            'SGD': SGDClassifier(random_state=random_state),
            'MLP': MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
            'SVM': SVC(kernel="linear", C=0.025, random_state=random_state, probability=True),
            'GaussianNB': GaussianNB(),
            'Perceptron': Perceptron(),
            'QDA': QuadraticDiscriminantAnalysis(),
            'GPClassifier': GaussianProcessClassifier(1.0 * RBF(1.0)),
            'KNN1': KNeighborsClassifier(1),
            'KNN3': KNeighborsClassifier(3),
            'KNN5': KNeighborsClassifier(5),
        }

    def set_comparison_classifiers(self, n_estimators, random_state):
        self.comparison_classifiers = {
            'XGBoost': XGBClassifier(n_estimators=n_estimators, max_depth=self.max_depth, random_state=random_state, n_jobs=1),
            'GBoost': GradientBoostingClassifier(n_estimators=n_estimators,  max_depth=self.max_depth, random_state=random_state),
            'RF': RandomForestClassifier(n_estimators=n_estimators,  max_depth=self.max_depth, random_state=random_state, n_jobs=1),
            'DT': DecisionTreeClassifier(max_depth=self.max_depth,  random_state=random_state)
        }

    def classifier_selection(self,width):
        self.main_clf = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_sample,
                                               random_state=width)

    def classifier_selection_old(self, width):
        clf = self.selected_classifier
        name = clf
        if clf == 'DT':
            name += str(self.max_depth)
        for i in range(self.no_classifiers):
            self.classifier_models[clf + str(i + 1)] = self.possible_classifiers(i+width, name)[name]  # TODO --> ASK KOSTAS ABOUT RANDOM STATE
            self.classifier_dict[i] = clf + str(i + 1)

