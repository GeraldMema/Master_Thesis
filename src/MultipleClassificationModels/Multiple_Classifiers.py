
from sklearn.model_selection import cross_val_score
from scipy import stats as s
from sklearn.metrics import f1_score


class Multiple_Classifiers:
    """
    Add Description
    """

    def __init__(self, multiple_classification_models_params, classifiers):
        self.classifiers = classifiers
        self.predictions_per_classifier = {}
        self.scores = {}
        self.predictions = []
        self.fusion_method = multiple_classification_models_params['fusion_method']
        self.evaluation_metric = multiple_classification_models_params['fitness_score_metric']

    def fit(self, X_train, y_train, clf):
        """
        Add description
        """
        return self.classifiers.classifier_models[clf].fit(X_train, y_train)

    def predict(self, X_test, clf_model, clf):
        """
        Add description
        """
        self.predictions_per_classifier[clf] = clf_model.predict(X_test)

    def score_additional(self, X, y, clf_model, clf, scoring_method='f1_micro', cv=5):
        """
        Add description
        """
        all_scores = cross_val_score(clf_model, X, y, scoring=scoring_method, cv=cv)
        self.scores[clf] = all_scores.mean()

    def score(self, y_cv, clf):
        """
        Add description
        """
        if self.evaluation_metric == 'f1_micro':
            self.scores[clf] = f1_score(y_cv, self.predictions_per_classifier[clf], average='micro')

    def predict_ensemble(self, m):
        """
        Add description
        """
        self.predictions = []
        if self.fusion_method == 'majority_voting':
            for i in range(m):
                guess = []
                for clf in self.predictions_per_classifier:
                    guess.append(self.predictions_per_classifier[clf][i])
                y_hat = s.mode(guess)[0][0]
                self.predictions.append(y_hat)
