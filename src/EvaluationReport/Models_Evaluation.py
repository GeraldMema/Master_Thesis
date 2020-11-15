from src.MultipleClassificationModels.Classifiers import Classifiers
from src.MultipleClassificationModels.Multiple_Classifiers import Multiple_Classifiers
from sklearn.metrics import f1_score

import time


class Models_Evaluation:

    def __init__(self, evaluation_params):
        self.score_metric = evaluation_params['score_metric']

    def get_score(self, y_test, y_hat, ):
        if self.score_metric == 'f1_micro':
            return f1_score(y_test, y_hat, average='micro')

    def my_alg_evalution(self, train_data_per_classifier, test_data_per_classifier, y_train, y_test, params):
        start_time = time.time()

        mc = Multiple_Classifiers(params, Classifiers(params))
        # Produce the final results
        for clf in train_data_per_classifier:
            model = mc.fit_1D(train_data_per_classifier[clf], y_train)
            mc.predict_per_solution(test_data_per_classifier[clf], model, clf)
        m = len(y_test)
        mc.predict_ensemble(m, True, y_test)

        # My Algorithm
        final_score = self.get_score(y_test, mc.predictions)
        stop = round((time.time() - start_time), 2)

        return final_score, stop, mc

    def other_evaluation(self, model, X_train, y_train, X_test, y_test):
        start_time = time.time()
        model.fit(X_train, y_train)
        score = self.get_score(y_test, model.predict(X_test))
        stop = round((time.time() - start_time), 2)
        return score, stop
