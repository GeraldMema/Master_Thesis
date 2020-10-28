import numpy as np
from random import randrange


class Population:
    """
    Add Description
    """

    def __init__(self, population_size, solution_representation):
        self.max_population_size = population_size
        self.solution_representation = solution_representation
        self.current_pop = None

    def random_initialization(self):
        """
        Initialize random solutions. The number of solutions is equal or less(we will keep only the unique solutions)
        to the max population size size.
        The result of initial solution is a list of chromosomes. The chromosome structure depends
        on the representation method we used
        :param
        :return:
        """

        initial_population = []

        max_no_selected_classifiers = randrange(self.solution_representation.solution_info.no_classifiers)
        max_no_selected_features = randrange(self.solution_representation.solution_info.no_features)
        if self.solution_representation == '1D':
            for _ in range(self.max_population_size):
                # select random the classifiers and features
                selected_classifiers = [randrange(self.solution_representation.solution_info.no_classifiers) for _ in range(max_no_selected_classifiers)]
                selected_features = [randrange(self.solution_representation.solution_info.no_features) for _ in range(max_no_selected_features)]
                # 1D representation function
                self.solution_representation.oneD_representation(selected_classifiers, selected_features)
                # add chromosome to the population
                initial_population.append(self.solution_representation.chromosome)
        elif self.solution_representation == '2D':
            for _ in range(self.max_population_size):
                feat_per_clf = []
                # for each classifier select random the features
                for c in self.solution_representation.solution_info.no_classifiers:
                    selected_features = [randrange(self.solution_representation.solution_info.no_features) for _ in
                                         range(max_no_selected_features)]
                    feat_per_clf.append(selected_features)
                # 2D representation function
                self.solution_representation.twoD_representation(feat_per_clf)
                # add chromosome to the population
                initial_population.append(self.solution_representation.chromosome)
        elif self.solution_representation == 'dual':
            print("TODO")

        # convert all solutions to a numpy array
        init_population_array = np.stack(initial_population, axis=0)
        # keep only the unique values
        # !!! WARNING : Maybe the initial solutions will not be equal to the populations
        self.current_pop = np.unique(init_population_array, axis=0)

    def opposition_based_learning_initialization(self):
        """
        Add Description
        :param
        :return:
        """
        pass
