class Solution_Info:
    """
    Add Description
    """

    def __init__(self, solution, population_producer):
        self.chromosome = solution
        self.features_per_classifiers = None
        self.fitness_score = -1
        self.diversity_score = -1
        self.accuracy_score = -1
        self.best_fusion_method = None
        self.population_producer = population_producer
        self.shape = solution.shape
        self.prediction_diversity = -1
        self.prediction_accuracy = -1
        self.accuracy_ens_score = -1


