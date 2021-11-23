from sklearn.metrics import f1_score,accuracy_score



import time


class Models_Evaluation:

    def __init__(self, evaluation_params):
        self.score_metric = evaluation_params['score_metric']
        self.des_data_size = evaluation_params['des_data_size']

    def get_score(self, y_test, y_hat, ):
        if self.score_metric == 'f1_micro':
            return f1_score(y_test, y_hat, average='micro')
        elif self.score_metric == 'accuracy':
            return accuracy_score(y_test, y_hat)

    def my_alg_evalution(self, train_data_per_classifier, test_data_per_classifier, y_train, y_test, mc):
        start_time = time.time()
        preds = []

        # Produce the final results
        for clf_idx in train_data_per_classifier:
            model = mc.fit(train_data_per_classifier[clf_idx], y_train)
            mc.predict_per_classifier(test_data_per_classifier[clf_idx], model, clf_idx)
            preds.append(mc.predictions_per_classifier[clf_idx])
        m = len(y_test)
        mc.predict_ensemble(m, True, y_test)

        # My Algorithm
        final_score = self.get_score(y_test, mc.predictions_ens)
        stop = round((time.time() - start_time), 2)

        gatg_div = None
        sum_div = 0
        y_ens = list(mc.predictions_ens)
        for dt_clf_pred in preds:
            y_clf = list(dt_clf_pred)
            diff_preds = sum([1 if x != y_clf[i] else 0 for i, x in enumerate(y_ens)])
            sum_div += diff_preds / len(y_ens)
        gatg_div = sum_div/len(preds)

        return final_score, stop, mc, gatg_div

    def other_evaluation(self, model, X_train, y_train, X_test, y_test, isRF=False):
        start_time = time.time()
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        score = self.get_score(y_test, y_hat)
        stop = round((time.time() - start_time), 2)
        rf_div = None
        if isRF:
            rf_clfs = model.estimators_
            no_clfs = len(rf_clfs)
            sum_div = 0
            for cl in rf_clfs:
                dt_pred = list(cl.predict(X_test))
                rf_pred = list(y_hat)
                diff_preds = sum([1 if x != dt_pred[i] else 0 for i, x in enumerate(rf_pred)])
                sum_div += diff_preds/len(rf_pred)
            rf_div = sum_div/no_clfs


        return score, stop, rf_div

