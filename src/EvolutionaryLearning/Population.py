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

    def __init__(self, evolutionary_learning_params, evolutionary_learning_methods, solution_representation):
        self.max_population_size = evolutionary_learning_params['population_size']
        self.solution_representation = solution_representation
        self.current_pop = None
        self.crossover_pop = None
        self.mutation_pop = None
        self.evolutionary_learning_methods = evolutionary_learning_methods
        self.evolutionary_learning_params = evolutionary_learning_params

    def random_initialization(self):
        """
        Initialize random solutions. The number of solutions is equal or less(we will keep only the unique solutions)
        to the max population size size.
        The result of initial solution is a list of chromosomes. The chromosome structure depends
        on the representation method we used
        :param
        :return:
        """

        global selected_features
        initial_population = []


        if self.solution_representation.representation_method == '1D':
            for _ in range(self.max_population_size):
                # select random the features
                no_features = True
                while no_features:
                    max_no_selected_features = randrange(self.solution_representation.max_no_features)
                    selected_features = [randrange(self.solution_representation.max_no_features) for _ in
                                         range(max_no_selected_features)]
                    if len(selected_features) > 0:
                        no_features = False
                # 1D representation function
                self.solution_representation.oneD_representation(selected_features)
                # add chromosome to the population
                initial_population.append(self.solution_representation.chromosome)
        elif self.solution_representation.representation_method == '2D':
            for _ in range(self.max_population_size):
                feat_per_clf = []
                # for each classifier select random the features
                for c in range(self.solution_representation.max_no_classifiers):
                    no_features = True
                    while no_features:
                        max_no_selected_features = randrange(self.solution_representation.max_no_features)
                        selected_features = [randrange(self.solution_representation.max_no_features) for _ in
                                             range(max_no_selected_features)]
                        if len(selected_features) > 0:
                            no_features = False
                    feat_per_clf.append(selected_features)
                # 2D representation function
                self.solution_representation.twoD_representation(feat_per_clf)
                # add chromosome to the population
                initial_population.append(self.solution_representation.chromosome)
        elif self.solution_representation.representation_method == 'dual':
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

    def generate_crossover_population(self, solution_dict, fitness_values, nc):
        k = 0
        offspring = []
        valid_pop = False
        while k < nc:
            # select parents
            parent_selection = Parent_Selection(solution_dict, fitness_values, self.evolutionary_learning_methods)
            p1 = list(parent_selection.mate[0])
            p2 = list(parent_selection.mate[1])
            # apply crossover operator
            crossover = Crossover_Operator(p1, p2, self.evolutionary_learning_methods)
            offspring.append(crossover.offspring_1)
            offspring.append(crossover.offspring_2)
            k += 2
        self.crossover_pop = np.unique(np.stack(offspring, axis=0), axis=0)

    def generate_mutation_population(self, solution_dict, nm):
        # mutation phase
        m = 0
        mutants = []
        while m < nm:
            mutant_idx = random.choice(list(solution_dict.keys()))
            parent = solution_dict[mutant_idx]
            mutation = Mutation_Operator(parent, self.evolutionary_learning_methods, self.evolutionary_learning_params)
            if mutation.mutant not in mutants:
                mutants.append(mutation.mutant)
                m += 1
        self.mutation_pop = np.unique(np.stack(mutants, axis=0), axis=0)
