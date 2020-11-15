import numpy as np
import math
import random


def bit_flip_mutation(mutant_array, mutation_rate, representation_method):
    # select parent
    if representation_method == '1D':
        parent = list(mutant_array)
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
    elif representation_method == '2D':
        mutant_list = [list(m) for m in mutant_array]
        mutant_list_return = []
        for parent in mutant_list:
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
            mutant_list_return.append(mutant)
        return mutant_list_return
    elif representation_method == 'dual':
        return None  # TODO
    else:
        return None


def bit_string_mutation(mutant_array, mutation_rate, representation_method):
    # initialize the mutant
    if representation_method == '1D':
        m = list(mutant_array)
        mutant = m.copy()
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
    elif representation_method == '2D':
        mutant_list = [list(m) for m in mutant_array]
        mutant_list_return = []
        for m in mutant_list:
            mutant = m.copy()
            # select randomly the number of genes to mutate
            number_of_genes_to_mutate = math.ceil(mutation_rate * len(mutant))
            genes = random.sample(range(0, len(mutant)), number_of_genes_to_mutate)
            # mutation phase
            for gene in genes:
                if mutant[gene] == 0:
                    mutant[gene] = 1
                else:
                    mutant[gene] = 0
            mutant_list_return.append(mutant)
        return mutant_list_return
    elif representation_method == 'dual':
        mutant = None  # TODO
        return mutant
    else:
        mutant = None
        return mutant


class Mutation_Operator:
    def __init__(self, parent, evolutionary_learning_methods, evolutionary_learning_params):
        mutation_method = evolutionary_learning_methods['mutation_methods']
        mutation_rate = evolutionary_learning_params['mutation_rate']
        representation_method = evolutionary_learning_methods['chromosome_representation']
        valid_mutant = False
        while not valid_mutant:
            if mutation_method == "Bit String Mutation":
                mutant = bit_string_mutation(parent, mutation_rate, representation_method)
            else:
                mutant = bit_flip_mutation(parent, mutation_rate, representation_method)
            if representation_method == '1D':
                if sum(mutant) != 0:
                    self.mutant = mutant
                    valid_mutant = True
            elif representation_method == '2D':
                is_zero = False
                for m in mutant:
                    if sum(m) == 0:
                        is_zero = True
                        break
                if not is_zero:
                    self.mutant = mutant
                    valid_mutant = True
            elif representation_method == 'dual':
                #TODO
                self.mutant = None
            else:
                self.mutant = None