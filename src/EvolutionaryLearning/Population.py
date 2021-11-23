import numpy as np
import random
import pandas as pd
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
        self.init_pop = None
        self.mutation_pop = None
        self.evolutionary_learning_methods = evolutionary_learning_methods
        self.evolutionary_learning_params = evolutionary_learning_params
        self.no_features = no_features
        self.no_classifiers = no_classifiers
        self.method = evolutionary_learning_methods['initialization_method']
        if self.method == 'random_initialization':
            self.random_initialization()
        elif self.method == 'only_ones':
            self.only_ones()
        self.pop_stats = {}

    def random_initialization(self):
        """
        Initialize random solutions. The number of solutions is equal or less(we will keep only the unique solutions)
        to the max population size size.
        The result of initial solution is a list of chromosomes. The chromosome structure depends
        on the representation method we used
        :param
        :return:
        """

        initial_population = {}
        start_idx = 0
        for _ in range(2 * self.max_population_size):
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
            initial_population[start_idx] = chromosome
            start_idx += 1

        self.current_pop = pd.DataFrame.from_dict(initial_population, orient='index').drop_duplicates().head(self.max_population_size)

    def get_no_classifier_per_feature(self):
        classifiers_per_features = self.current_pop.sum(axis=0)
        start_classifiers_per_features = {}
        for i in range(len(classifiers_per_features)):
            start_classifiers_per_features[i] = classifiers_per_features[i]
        return start_classifiers_per_features

    def opposition_based_learning_initialization(self):
        """
        Add Description
        :param
        :return:
        """
        pass

    def generate_crossover_population(self, population, fitness_values, nc):
        valid_pop = False
        start_idx = max(fitness_values, key=int) + 1

        k = 0
        offspring = {}
        while k < nc:
            # select parents
            parent_selection = Parent_Selection(population, fitness_values,
                                                self.evolutionary_learning_methods)
            p1 = list(parent_selection.mate[0])
            p2 = list(parent_selection.mate[1])

            # apply crossover operator
            crossover = Crossover_Operator(p1, p2, self.evolutionary_learning_methods)
            offspring[start_idx] = crossover.offspring_1
            start_idx += 1
            offspring[start_idx] = crossover.offspring_2
            start_idx += 1
            k += 2

        self.crossover_pop = pd.DataFrame.from_dict(offspring, orient='index').drop_duplicates()

    def generate_mutation_population(self, population, nm):
        m = 0
        mutants = {}
        start_idx = population.crossover_pop.index.max() + 1
        while m < nm:
            mutant_idx = random.choice(list(population.current_pop.index))
            parent = population.current_pop.loc[mutant_idx]
            mutation = Mutation_Operator(parent, self.evolutionary_learning_methods,
                                         self.evolutionary_learning_params)
            # if mutation.mutant not in mutants:
            mutants[start_idx] = mutation.mutant
            m += 1
            start_idx += 1
        self.mutation_pop = pd.DataFrame.from_dict(mutants, orient='index').drop_duplicates()

    def only_ones(self):
        """
        Add Description here

        """
        initial_population = {}

        for i in range(self.max_population_size):
            chromosome = np.ones(self.no_features)
            initial_population[i] = chromosome
        self.current_pop = pd.DataFrame.from_dict(initial_population, orient='index')
