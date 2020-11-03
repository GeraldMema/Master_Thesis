class Fitness_Evaluation:
    """
    Add Description
    """

    def __init__(self, evolutionary_learning_params, multiple_classifiers):
        self.w = {}
        self.alpha = evolutionary_learning_params['fitness_lambda']
        self.o = multiple_classifiers.predictions_per_classifier
        self.oe = multiple_classifiers.predictions
        self.scores = multiple_classifiers.scores
        self.init_weights_of_classifiers()
        self.fitness_value = 0

    def init_weights_of_classifiers(self):
        no_classifiers = len(self.o)
        for clf in self.o:
            self.w[clf] = 1 / no_classifiers

    def fitness_function(self):
        """
        Add Description
        TODO Implement the cost function of this paper
        Generating Accurate and Diverse Members of a Neural-Network Ensemble
        Accuracy + λ*   Diversity = f1 + λ*Di ,
        :param
        :return:
        """

        Di, D = self.diversity()
        Ei, E = self.error()

        fitness = (1 - E) + self.alpha * D
        self.fitness_value = fitness

    def error(self):
        """
        Add Description
        Di = Σ_x [Oi(X) - o'(x)]^2

        """
        Ei = {}
        E = 0
        for clf in self.scores:
            Ei[clf] = 1 - self.scores[clf]
            E = E + (Ei[clf] * self.w[clf])
        return Ei, E

    def diversity(self):
        """
        Add Description
        Di = Σ_x [Oi(X) - o'(x)]^2

        """
        Di = {}
        D = 0
        for clf in self.o:
            sum = 0
            for i in range(len(self.o[clf])):
                sum = sum + (self.o[clf][i] == self.oe[i]) # Change from paper (classification instead of regression)
            Di[clf] = sum / len(self.o[clf])  # Change from paper (Normalize)
            D = D + (Di[clf] * self.w[clf])
        return Di, D
