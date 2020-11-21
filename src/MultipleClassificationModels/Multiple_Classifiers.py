from scipy import stats as s
from sklearn.metrics import f1_score


class Multiple_Classifiers:
    """
    Add Description
    """

    def __init__(self, multiple_classification_models_params, classifiers):
        self.classifiers = classifiers
        self.predictions_per_classifier = {}
        self.score_per_classifier = {}
        self.score_ens = -1
        self.predictions_ens = []
        self.fusion_method = multiple_classification_models_params['fusion_method']
        self.evaluation_metric = multiple_classification_models_params['fitness_score_metric']
        self.cross_validation = multiple_classification_models_params['cross_val']
        self.useful_info = {}

    def fit(self, X_train, y_train, clf_idx):
        """
        Add description
        """
        clf_name = self.classifiers.classifier_dict[clf_idx]
        return self.classifiers.classifier_models[clf_name].fit(X_train, y_train)

    def predict_per_classifier(self, X_test, clf_model, clf_idx):
        """
        Add description
        """
        self.predictions_per_classifier[clf_idx] = clf_model.predict(X_test)

    def predict_ensemble(self, m, is_final=False, y_test=None):
        """
        Add description
        """
        if self.fusion_method == 'majority_voting':
            for i in range(m):
                guess = []
                for classifier_idx in self.predictions_per_classifier:
                    guess.append(self.predictions_per_classifier[classifier_idx][i])
                y_hat = s.mode(guess)[0][0]  # we can extract and the number of voting
                if is_final:
                    self.useful_info[i] = set(guess), y_hat, y_test.iloc[i]
                self.predictions_ens.append(y_hat)

    def get_score_per_classifier(self, y_cv, classifier_idx):
        """
        Add description
        """
        if self.evaluation_metric == 'f1_micro':
            self.score_per_classifier[classifier_idx] = f1_score(y_cv, self.predictions_per_classifier[classifier_idx], average='micro')
        elif self.evaluation_metric == 'f1_macro':
            self.score_per_classifier[classifier_idx] = f1_score(y_cv, self.predictions_per_classifier[classifier_idx], average='macro')

    def get_score_ensemble(self, y_cv):
        """
        Add description
        """
        if self.evaluation_metric == 'f1_micro':
            self.score_ens = f1_score(y_cv, self.predictions_ens, average='micro')
        elif self.evaluation_metric == 'f1_macro':
            self.score_ens = f1_score(y_cv, self.predictions_ens, average='macro')

