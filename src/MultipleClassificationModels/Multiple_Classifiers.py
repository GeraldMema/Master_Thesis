from scipy import stats as s
from sklearn.metrics import f1_score,accuracy_score


class Multiple_Classifiers:
    """
    Add Description
    """

    def __init__(self, multiple_classification_models_params, classifiers):
        self.classifiers = classifiers
        self.predictions_per_classifier = {}
        self.predictions_probs_per_classifier = {}
        self.score_per_classifier = {}
        self.score_ens = -1
        self.predictions_ens = []
        self.fusion_method = multiple_classification_models_params['fusion_method']
        self.evaluation_metric = multiple_classification_models_params['fitness_score_metric']
        self.cross_validation = multiple_classification_models_params['cross_val']
        self.useful_info = {}
        self.max_clf = 10

    def fit(self, X_train, y_train):
        """
        Add description
        """

        return self.classifiers.main_clf.fit(X_train, y_train)

    def fit_old(self, X_train, y_train, clf_idx):
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
        self.predictions_probs_per_classifier[clf_idx] = clf_model.predict_proba(X_test)

    def predict_ensemble(self, m, is_final=False, y_test=[]):
        """
        Add description
        """
        if self.fusion_method == 'majority_voting':
            for i in range(m):
                guess = []
                count_per_guess = {}
                for classifier_idx in self.predictions_per_classifier:

                    pred = self.predictions_per_classifier[classifier_idx][i]
                    guess.append(pred)
                    if pred not in count_per_guess:
                        count_per_guess[pred] = 1
                    else:
                        count_per_guess[pred] += 1
                y_hat = s.mode(guess)[0][0]  # we can extract and the number of voting
                if is_final:
                    self.useful_info[i] = count_per_guess, y_hat, y_test.iloc[i]
                self.predictions_ens.append(y_hat)
        elif self.fusion_method == 'weighted_sum':
            for i in range(m):
                guess = []
                count = 1

                for classifier_idx in self.predictions_per_classifier:
                    probs = self.predictions_probs_per_classifier[classifier_idx][i]
                    guess.append(probs)

                transpose_guess = list(map(list, zip(*guess)))
                # print(transpose_guess)
                weighted_sum = [sum(x) for x in transpose_guess]
                y_hat = weighted_sum.index(max(weighted_sum))
                # if any(y_test):
                #     if i%10==0:
                #         print('Instance {}/{} ---> Results: {}-{}/{}'.format(i,m, y_test.iloc[i],y_hat, transpose_guess))
                self.predictions_ens.append(y_hat)

    def get_score_per_classifier(self, y_cv, classifier_idx):
        """
        Add description
        """
        if self.evaluation_metric == 'f1_micro':
            self.score_per_classifier[classifier_idx] = f1_score(y_cv, self.predictions_per_classifier[classifier_idx],
                                                                 average='micro')
        elif self.evaluation_metric == 'f1_macro':
            self.score_per_classifier[classifier_idx] = f1_score(y_cv, self.predictions_per_classifier[classifier_idx],
                                                                 average='macro')
        elif self.evaluation_metric == 'accuracy':
            self.score_per_classifier[classifier_idx] = accuracy_score(y_cv, self.predictions_per_classifier[classifier_idx])

    def get_score_ensemble(self, y_cv):
        """
        Add description
        """

        if self.evaluation_metric == 'f1_micro':
            self.score_ens = f1_score(y_cv, self.predictions_ens, average='micro')
        elif self.evaluation_metric == 'f1_macro':
            self.score_ens = f1_score(y_cv, self.predictions_ens, average='macro')
        elif self.evaluation_metric == 'accuracy':
            self.score_ens = accuracy_score(y_cv, self.predictions_ens)
