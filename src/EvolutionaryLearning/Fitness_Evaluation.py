class Fitness_Evaluation:
    """
    Add Description
    """

    def __init__(self, evolutionary_learning_params, multiple_classifiers, evolutionary_learning_methods, y_true):
        self.w = {}
        self.alpha = evolutionary_learning_params['fitness_lambda']
        self.oe = multiple_classifiers.predictions
        self.o = multiple_classifiers.predictions_per_solution
        self.score_ens = multiple_classifiers.score_ens
        self.scores = multiple_classifiers.scores
        self.predictions_size = len(self.oe)
        self.no_classifiers = len(self.o)
        self.init_weights_of_classifiers()
        if evolutionary_learning_methods['diversity_only_error']:
            self.only_error = True
        else:
            self.only_error = False
        if evolutionary_learning_methods['chromosome_representation'] == '2D':
            self.fitness_value = 0
            self.diversity_value = 0
            self.accuracy_value = 0
            self.fitness_ens(y_true.tolist())
        elif evolutionary_learning_methods['chromosome_representation'] == '1D':
            self.fitness_value = {}
            self.diversity_value = {}
            self.accuracy_value = {}
            self.fitness_clf(y_true.tolist())

    def init_weights_of_classifiers(self):
        for clf in range(self.no_classifiers):
            self.w[clf] = 1 / self.no_classifiers

    def fitness_clf(self, y_true):
        """
        Add Description

        :param
        :return
        """

        for solution_idx in self.o:
            Di = self.diversity_per_classifier(solution_idx, y_true)
            self.calc_fitness_per_classifier(Di, solution_idx)

    def fitness_ens(self, y_true):
        """
        Add Description

        :param
        :return
        """
        self.diversity_value = self.diversity(y_true)
        self.accuracy_value = self.score_ens
        self.fitness_value = ((1 - self.alpha) * self.accuracy_value) + (self.alpha * self.diversity_value)

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
                    different_guess = different_guess + 1
                else:  # We focus only on the classifiers error or ensemble error ??????
                    if y_true[i] != self.oe[i]:
                        wrong_predictions = wrong_predictions + 1
                        different_guess = different_guess + 1
            else:
                if y_true[i] != self.oe[i]:
                    wrong_predictions = wrong_predictions + 1
        if self.only_error:
            count = wrong_predictions
            if count == 0:
                return 1  # No wrong predictions
        else:
            count = self.predictions_size

        Di = different_guess / count  # Change from paper (Normalize)

        return Di

    def diversity(self, y_true):
        """
        Add Description
        Di = Σ_x [Oi(X) - o'(x)]^2

        """
        D = 0
        for clf in range(self.no_classifiers):
            D = D + (self.diversity_per_classifier(self, clf, y_true) * self.w[clf])
        return D

    def calc_fitness_per_classifier(self, Di, solution_idx):
        Ai = self.scores[solution_idx]
        self.fitness_value[solution_idx] = ((1 - self.alpha) * Ai) + (self.alpha * Di)
        self.accuracy_value[solution_idx] = Ai
        self.diversity_value[solution_idx] = Di
