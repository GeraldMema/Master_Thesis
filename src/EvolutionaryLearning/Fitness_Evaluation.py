class Fitness_Evaluation:
    """
    Add Description
    """

    def __init__(self, evolutionary_learning_params, multiple_classifiers, evolutionary_learning_methods, y_true,
                 alpha):
        self.w = {}
        self.oe = multiple_classifiers.predictions_ens
        self.o = multiple_classifiers.predictions_per_classifier
        self.A_ens = multiple_classifiers.score_ens
        self.Ai = multiple_classifiers.score_per_classifier
        self.Di = {}
        self.D_avg = -1
        self.A_avg = -1
        self.predictions_size = len(self.oe)
        self.no_classifiers = len(self.o)
        self.init_weights_of_classifiers()
        self.r = evolutionary_learning_params['lambda_factor']
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

    def diversity_per_classifier(self, solution_idx, y_true):
        """
        Add Description
        Di = Σ_x [Oi(X) - o'(x)]^2

        """
        if self.predictions_size == 0:
            print('Something went wrong with predictions')
            return
        different_guess = 0
        wrong_predictions = 0
        for i in range(self.predictions_size):
            if self.o[solution_idx][i] != self.oe[i]:
                if not self.only_error:
                    different_guess += 1
                else:  # We focus only on the classifiers error or ensemble error ??????
                    if y_true[i] != self.oe[i]:
                        wrong_predictions += 1
                        different_guess += 1
            else:
                if y_true[i] != self.oe[i]:
                    wrong_predictions += 1
        if self.only_error:
            count = wrong_predictions
        else:
            count = self.predictions_size

        epsilon = 1e-10
        self.Di[solution_idx] = different_guess / (count + epsilon)  # Change from paper (Normalize)

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
        self.fitness_value[solution_idx] = ((1 - alpha) * self.Ai[solution_idx]) + (alpha * self.Di[solution_idx])

    def adjust_lambda(self, previous_ens_accuracy, previous_mean_accuracy, previous_diversity, alpha):
        """

        1. we never change lambda if the ensemble error E is decreasing while we consider new networks;
        2. we change lambda if:
            a.population error E_ens is not increasing and the population diversity D_ens is decreasing;
                diversity seems to be under-emphasized and we increase lambda
            b. E_ens is increasing and D_ens is not decreasing; diversity seems to be over-emphasized and we decrease A
        """

        # Calc Mean Accuracy
        self.A_avg = self.get_accuracy_value()
        # Calc Mean Diversity
        self.D_avg = self.get_diversity_value()

        # We never change lambda if the ensemble error E is decreasing while
        if self.A_ens >= previous_ens_accuracy:
            return alpha
        else:
            # population error E_avg is not increasing and the population diversity D_avg is decreasing
            first_scenario = (self.A_avg >= previous_mean_accuracy) & (self.D_avg < previous_diversity)
            if first_scenario:
                alpha = (1 + self.r) * alpha  # increase the lambda

            # E_avg is increasing and D_ens is not decreasing; diversity seems to be over-emphasized and we decrease A
            second_scenario = (self.A_avg < previous_mean_accuracy) & (self.D_avg >= previous_diversity)
            if second_scenario:
                alpha = (1 - self.r) * alpha  # decrease the lambda

        if alpha > 1:
            alpha = 0.99

        return alpha
