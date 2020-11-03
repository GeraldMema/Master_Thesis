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
        self.parent_selection_method = evolutionary_learning_methods['parent_selection_method']
        self.crossover_method = evolutionary_learning_methods['crossover_methods']
        self.mutation_method = evolutionary_learning_methods['mutation_methods']
        self.mutation_rate = evolutionary_learning_params['mutation_rate']

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

        max_no_selected_classifiers = randrange(self.solution_representation.max_no_classifiers)
        max_no_selected_features = randrange(self.solution_representation.max_no_features)
        if self.solution_representation.representation_method == '1D':
            for _ in range(self.max_population_size):
                # select random the classifiers and features
                selected_classifiers = [randrange(self.solution_representation.max_no_classifiers) for _ in
                                        range(max_no_selected_classifiers)]
                selected_features = [randrange(self.solution_representation.max_no_features) for _ in
                                     range(max_no_selected_features)]
                # 1D representation function
                self.solution_representation.oneD_representation(selected_classifiers, selected_features)
                # add chromosome to the population
                initial_population.append(self.solution_representation.chromosome)
        elif self.solution_representation.representation_method == '2D':
            for _ in range(self.max_population_size):
                feat_per_clf = []
                # for each classifier select random the features
                for c in range(self.solution_representation.max_no_classifiers):
                    selected_features = [randrange(self.solution_representation.max_no_features) for _ in
                                         range(max_no_selected_features)]
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
        while k < nc:
            # select parents
            parent_selection = Parent_Selection(solution_dict, fitness_values, self.parent_selection_method)
            p1 = list(parent_selection.mate[0])
            p2 = list(parent_selection.mate[1])
            # apply crossover operator
            crossover = Crossover_Operator(p1, p2, self.crossover_method,
                                           self.solution_representation.chromosome_length)
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
            mutation = Mutation_Operator(parent, self.mutation_rate, self.mutation_method,
                                         self.solution_representation.representation_method)
            if mutation.mutant not in mutants:
                mutants.append(mutation.mutant)
                m += 1
        self.mutation_pop = np.unique(np.stack(mutants, axis=0), axis=0)


