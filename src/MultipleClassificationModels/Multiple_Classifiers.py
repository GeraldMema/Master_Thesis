from sklearn.model_selection import cross_val_score
from scipy import stats as s
from sklearn.metrics import f1_score


class Multiple_Classifiers:
    """
    Add Description
    """

    def __init__(self, multiple_classification_models_params, classifiers):
        self.classifiers = classifiers
        self.predictions_per_solution = {}
        self.scores = {}
        self.scores_per_solution = {}
        self.score_ens = -1
        self.predictions = []
        self.fusion_method = multiple_classification_models_params['fusion_method']
        self.evaluation_metric = multiple_classification_models_params['fitness_score_metric']
        self.cross_validation = multiple_classification_models_params['cross_val']
        self.useful_info = {}

    def fit(self, X_train, y_train, solution_idx):
        """
        Add description
        """

        return self.classifiers.classifier_models[solution_idx].fit(X_train, y_train)

    def fit_1D(self, X_train, y_train):
        """
        Add description
        """
        clf = self.classifiers.classifier_dict[0]
        return self.classifiers.classifier_models[clf].fit(X_train, y_train)

    def predict_per_solution(self, X_test, clf_model, solution_idx):
        """
        Add description
        """
        self.predictions_per_solution[solution_idx] = clf_model.predict(X_test)

    def score_additional(self, X, y, clf_model, solution_idx):
        """
        Add description
        """
        cv = self.cross_validation
        all_scores = cross_val_score(clf_model, X, y, scoring=self.evaluation_metric, cv=cv)
        self.scores[solution_idx] = all_scores.mean()

    def score_per_solution(self, y_cv, solution_idx):
        """
        Add description
        """
        if self.evaluation_metric == 'f1_micro':
            self.scores[solution_idx] = f1_score(y_cv, self.predictions_per_solution[solution_idx], average='micro')
        elif self.evaluation_metric == 'f1_macro':
            self.scores[solution_idx] = f1_score(y_cv, self.predictions_per_solution[solution_idx], average='macro')

    def score_ensemble(self, y_cv):
        """
        Add description
        """
        if self.evaluation_metric == 'f1_micro':
            self.score_ens = f1_score(y_cv, self.predictions, average='micro')
        elif self.evaluation_metric == 'f1_macro':
            self.score_ens = f1_score(y_cv, self.predictions, average='macro')

    def predict_ensemble(self, m, is_final=False, y_test=None):
        """
        Add description
        """
        self.predictions = []
        if self.fusion_method == 'majority_voting':
            for i in range(m):
                guess = []
                for solution_idx in self.predictions_per_solution:
                    guess.append(self.predictions_per_solution[solution_idx][i])
                y_hat = s.mode(guess)[0][0]  # we can extract and the number of voting
                if is_final:
                    self.useful_info[i] = set(guess), y_hat, y_test.iloc[i]
                self.predictions.append(y_hat)
