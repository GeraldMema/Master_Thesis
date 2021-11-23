import numpy as np
class Fitness_Evaluation:
    """
    Add Description
    """

    def __init__(self, evolutionary_learning_params, multiple_classifiers, evolutionary_learning_methods, y_true,
                 alpha):
        self.my_rules = evolutionary_learning_methods['my_rules']
        self.w = {}
        self.oe = multiple_classifiers.predictions_ens
        self.o = multiple_classifiers.predictions_per_classifier
        self.A_ens = multiple_classifiers.score_ens
        self.Ai = multiple_classifiers.score_per_classifier
        self.Di = {}
        self.D_avg = 1
        self.D_std = 1
        self.A_avg = 1
        self.A_std = 1
        self.F_avg = 1
        self.predictions_size = len(self.oe)
        self.no_classifiers = len(self.o)
        self.init_weights_of_classifiers()
        self.r = evolutionary_learning_params['lambda_factor']
        self.lambda_factor_strategy = evolutionary_learning_methods['lambda_factor_strategy']
        self.one_minus_lambda = evolutionary_learning_methods['one_minus_lambda']
        if evolutionary_learning_methods['diversity_only_error']:
            self.only_error = True
        else:
            self.only_error = False
        self.fitness_value = {}
        self.fitness(y_true.tolist(), alpha)

    def init_weights_of_classifiers(self):
        for clf in range(self.no_classifiers):
            self.w[clf] = 1 / self.no_classifiers

    def fitness(self, y_true, alpha):
        """
        Add Description

        :param
        :return
        """

        for solution_idx in self.o:
            self.diversity_per_classifier(solution_idx, y_true)
            self.calc_fitness_per_classifier(solution_idx, alpha)

        # Calc Mean STD Diversity
        self.D_avg = self.get_diversity_value()
        self.D_std = np.std(list(self.Di.values()))
        # Calc Mean STD Accuracy
        self.A_avg = self.get_accuracy_value()
        self.A_std = np.std(list(self.Ai.values()))
        # Calc Mean Fitness
        self.F_avg = self.get_fitness_avg()

    def get_fitness_avg(self):
        s = 0
        for f in self.fitness_value:
            s += self.fitness_value[f]
        return s / len(self.fitness_value)

    def diversity_per_classifier(self, solution_idx, y_true):
        """
        Add Description
        Di = Σ_x [Oi(X) - o'(x)]^2

        """
        if self.predictions_size == 0:
            print('Something went wrong with predictions')
            return

        epsilon = 1e-5

        y_ens = list(self.oe)
        y_clf = list(self.o[solution_idx])
        if self.only_error:
            y = list(y_true)
            diversity_interest_idx = [i for i, x in enumerate(y) if x != y_ens[i]]
            new_y_ens = [y_ens[i] for i in diversity_interest_idx]
            new_y_clf = [y_clf[i] for i in diversity_interest_idx]
            diff_preds = sum([1 if x != new_y_clf[i] else 0 for i, x in enumerate(new_y_ens)])
            res = diff_preds/(len(new_y_ens) + epsilon)
        else:
            diff_preds = sum([1 if x != y_clf[i] else 0 for i, x in enumerate(y_ens)])
            res = diff_preds/(len(y_ens) + epsilon)

        self.Di[solution_idx] = res

    def get_diversity_value(self):
        D = 0
        for solution_idx in self.Di:
            D += self.Di[solution_idx]
        if len(self.Di) > 0:
            D /= len(self.Di)
        return D

    def get_accuracy_value(self):
        A = 0
        for solution_idx in self.Ai:
            A += self.Ai[solution_idx]
        if len(self.Ai) > 0:
            A /= len(self.Ai)
        return A

    def calc_fitness_per_classifier(self, solution_idx, alpha):
        if self.one_minus_lambda:
            self.fitness_value[solution_idx] = (((1 - alpha) * self.Ai[solution_idx]) + (alpha * self.Di[solution_idx]))
        else:
            self.fitness_value[solution_idx] = self.Ai[solution_idx] + (alpha * self.Di[solution_idx])

    def adjust_lambda(self, cache, alpha, A_gen, A):
        """

        1. we never change lambda if the ensemble error E is decreasing while we consider new networks;
        2. we change lambda if:
            a.population error E_ens is not increasing and the population diversity D_ens is decreasing;
                diversity seems to be under-emphasized and we increase lambda
            b. E_ens is increasing and D_ens is not decreasing; diversity seems to be over-emphasized and we decrease A
        """

        if len(cache) == 1:
            return alpha
        previous_ens_accuracy = cache[len(cache)-2][0]
        previous_mean_accuracy = cache[len(cache) - 2][1]
        previous_diversity = cache[len(cache) - 2][2]
        # We never change lambda if the ensemble error E is decreasing while
        if A_gen > previous_ens_accuracy:
            return alpha
        else:
            pr = previous_mean_accuracy
            coef = self.r
            if self.lambda_factor_strategy == 'mul':
                coef = self.r * alpha
            # population error E_avg is not increasing and the population diversity D_avg is decreasing
            first_scenario = (A >= pr) & (self.D_avg < previous_diversity)
            if first_scenario:
                alpha += coef  # increase the lambda

            # E_avg is increasing and D_ens is not decreasing; diversity seems to be over-emphasized and we decrease A
            second_scenario = (A < pr) & (self.D_avg >= previous_diversity)
            if second_scenario:
                alpha -= coef  # decrease the lambda

            if self.my_rules:

                third_scenario = (A >= pr) & (self.D_avg >= previous_diversity)
                if third_scenario:
                    alpha += coef  # decrease the lambda

                fourth_scenario = (A < pr) & (self.D_avg < previous_diversity)
                if fourth_scenario:
                    dif_ac = pr - A
                    dif_div = previous_diversity - self.D_avg
                    if dif_div > dif_ac:
                        alpha -= coef  # increase lambda
                    else:
                        alpha -= coef  # decrease lambda

        if alpha > 1:
            alpha = 0.99
        if alpha < 0:
            alpha = 0.01

        return alpha
