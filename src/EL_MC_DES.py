# Evolutionary Learning Using Multiple Classifiers and Dynamic Ensemble Selection
class EL_MC_DES:
    """
    Add Description
    """

    def __init__(self,
                 solver='lr',
                 no_classifiers=5,
                 population_initialization_method='random',
                 crossover_operator_method='two_point',
                 mutation_method='bit_flip_mutation',
                 parent_selection_method='roulette_wheel_selection',
                 fitness_evaluation_method='f1',
                 des_method='knn'
                 ):
        self.solver = solver
        self.no_classifiers = no_classifiers
        self.population_initialization_method = population_initialization_method
        self.crossover_operator_method = crossover_operator_method
        self.mutation_method = mutation_method
        self.parent_selection_method = parent_selection_method
        self.fitness_evaluation_method = fitness_evaluation_method
        self.des_method = des_method

    def fit(self, X, y):
        pass

    def predict(self, y):
        pass