import numpy as np
import math
import random


def bit_flip_mutation(parent_population, mutation_rate):
    # select parent
    parents_labels = parent_population.index.to_list()
    parent_label = np.random.choice(parents_labels)
    parent = parent_population.loc[parent_label, :].values[:-1].astype("int64")
    mutant = []
    for i in range(len(parent)):
        r = np.random.uniform(0.0, 1.0)
        if r >= mutation_rate:
            mutant.append(parent[i])
        else:
            if parent[i] == 1:
                mutant.append(0)
            else:
                mutant.append(1)
    return mutant


def bit_string_mutation(parent_population, mutation_rate):
    # select parent
    parents_labels = parent_population.index.to_list()
    parent_label = np.random.choice(parents_labels)
    parent = parent_population.loc[parent_label, :].values[:-1].astype("int64")
    # initialize the mutant
    mutant = parent.copy()
    mutant = list(mutant)
    # select randomly the number of genes to mutate
    number_of_genes_to_mutate = math.ceil(mutation_rate * len(mutant))
    genes = random.sample(range(0, len(mutant)), number_of_genes_to_mutate)
    # mutation phase
    for gene in genes:
        if mutant[gene] == 0:
            mutant[gene] = 1
        else:
            mutant[gene] = 0
    return mutant


class Mutation_Operator:
    def __init__(self, parent_population, mutation_rate, mutation_method, population_size):
        self.mutation_operator = mutation_method
        self.population_size = population_size
        valid_mutant = False
        while not valid_mutant:
            if mutation_method == "Bit String Mutation":
                mutant = bit_string_mutation(parent_population, mutation_rate)
            else:
                mutant = bit_flip_mutation(parent_population, mutation_rate)
            if sum(mutant) != 0:
                self.mutant = mutant
                valid_mutant = True
