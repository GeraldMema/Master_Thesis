import numpy as np
import random
from random import randrange

from src.EvolutionaryLearning.Parent_Selection import Parent_Selection
from src.EvolutionaryLearning.Crossover_Operator import Crossover_Operator
from src.EvolutionaryLearning.Mutation_Operator import Mutation_Operator



class Population:
    """
    Add Description
    """

    def __init__(self, evolutionary_learning_params, evolutionary_learning_methods, no_features, no_classifiers):
        self.max_population_size = evolutionary_learning_params['population_size']
        self.current_pop = None
        self.crossover_pop = None
        self.mutation_pop = None
        self.evolutionary_learning_methods = evolutionary_learning_methods
        self.evolutionary_learning_params = evolutionary_learning_params
        self.no_features = no_features
        self.no_classifiers = no_classifiers

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

        for _ in range(self.max_population_size):
            # select random the features
            no_features = True
            selected_features = []
            while no_features:
                max_no_selected_features = randrange(self.no_features)
                selected_features = [randrange(self.no_features) for _ in
                                     range(max_no_selected_features)]
                if len(selected_features) > 0:
                    no_features = False
            # 1D representation function
            chromosome = np.zeros(self.no_features)
            for feat in selected_features:
                chromosome[feat] = 1

            # add chromosome to the population
            initial_population.append(chromosome)

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

    def generate_crossover_population(self, solution_info_dict, fitness_values, nc):
        k = 0
        offspring = []
        valid_pop = False
        while k < nc:
            # select parents
            parent_selection = Parent_Selection(solution_info_dict, fitness_values, self.evolutionary_learning_methods)
            p1 = list(parent_selection.mate[0])
            p2 = list(parent_selection.mate[1])
            # apply crossover operator
            crossover = Crossover_Operator(p1, p2, self.evolutionary_learning_methods)
            offspring.append(crossover.offspring_1)
            offspring.append(crossover.offspring_2)
            k += 2
        self.crossover_pop = np.unique(np.stack(offspring, axis=0), axis=0)

    def generate_mutation_population(self, solution_info_dict, nm):
        # mutation phase
        m = 0
        mutants = []
        while m < nm:
            mutant_idx = random.choice(list(solution_info_dict.keys()))
            parent = solution_info_dict[mutant_idx].chromosome
            mutation = Mutation_Operator(parent, self.evolutionary_learning_methods, self.evolutionary_learning_params)
            if mutation.mutant not in mutants:
                mutants.append(mutation.mutant)
                m += 1
        self.mutation_pop = np.unique(np.stack(mutants, axis=0), axis=0)
