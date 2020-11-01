from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from statistics import mode
from scipy import stats as s


def set_heterogeneous_classifiers():
    return {'DT': DecisionTreeClassifier(max_depth=5), 'SVM': SVC(gamma=2, C=1), 'KNN': KNeighborsClassifier(3)}


def set_classifiers():
    return {'DT1': DecisionTreeClassifier(max_depth=5), 'DT2': DecisionTreeClassifier(max_depth=10),
            'DT3': DecisionTreeClassifier(max_depth=15)}


class Multiple_Classifiers:
    """
    Add Description
    """

    def __init__(self, multiple_classification_models_params):
        if multiple_classification_models_params['heterogeneous_classifiers']:
            self.classifiers = set_heterogeneous_classifiers()
        else:
            self.classifiers = set_classifiers()
        self.predictions_per_classifier = {}
        self.scores = {}
        self.predictions = []
        self.fusion_method = multiple_classification_models_params['fusion_method']

    def fit(self, X_train, y_train, clf):
        return self.classifiers[clf].fit(X_train, y_train)

    def predict(self, X_test, clf_model, clf):
        self.predictions_per_classifier[clf] = clf_model.predict(X_test)

    def score(self, X, y, clf_model, clf, scoring_method='f1_micro', cv=5):
        all_scores = cross_val_score(clf_model, X, y, scoring=scoring_method, cv=cv)
        self.scores[clf] = all_scores.mean()

    def predict_ensemble(self, m):
        """
        Add description
        """

        if self.fusion_method == 'majority_voting':
            for i in range(m):
                guess = []
                for clf in self.predictions_per_classifier:
                    guess.append(self.predictions_per_classifier[clf][i])
                y_hat = s.mode(guess)[0]
                self.predictions.append(y_hat)
