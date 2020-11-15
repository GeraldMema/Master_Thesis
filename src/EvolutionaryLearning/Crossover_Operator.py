import numpy as np
import random


class Crossover_Operator:
    def __init__(self, p1, p2, evolutionary_learning_methods):
        self.parent_1 = p1
        self.parent_2 = p2
        self.crossover_operator = evolutionary_learning_methods['crossover_methods']
        rep = evolutionary_learning_methods['chromosome_representation']
        self.chromosome_length = len(p1)
        valid_offspring = False
        self.offspring_1 = None
        self.offspring_2 = None
        while not valid_offspring:
            # produce offspring with respect to selected crossover operator
            if self.crossover_operator == 'single_point_crossover':
                offspring_1, offspring_2 = self.single_point_crossover()
            elif self.crossover_operator == 'two_point_crossover':
                offspring_1, offspring_2 = self.two_point_crossover()
            elif self.crossover_operator == 'three_point_crossover':
                offspring_1, offspring_2 = self.three_point_crossover()
            elif self.crossover_operator == 'uniform_crossover':
                if rep == '1D':
                    offspring_1, offspring_2 = self.uniform_crossover_1D()
                elif rep == '2D':
                    offspring_1, offspring_2 = self.uniform_crossover()
            elif self.crossover_operator == 'commonality_based_crossover':
                if rep == '1D':
                    offspring_1, offspring_2 = self.commonality_based_crossover_1D()
                elif rep == '2D':
                    offspring_1, offspring_2 = self.commonality_based_crossover()
            else:
                print('Wrong Crossover Methods')
                return
            # check offspring
            if rep == '1D':
                if (sum(offspring_1) != 0) and (sum(offspring_2) != 0):
                    self.offspring_1 = offspring_1
                    self.offspring_2 = offspring_2
                    valid_offspring = True
            elif rep == '2D':
                if (sum(offspring_1).any() != 0) and (sum(offspring_2).any() != 0):
                    self.offspring_1 = offspring_1
                    self.offspring_2 = offspring_2
                    valid_offspring = True

    def single_point_crossover(self):
        # avoid clones
        m = np.random.randint(1, self.chromosome_length - 2)
        # collect genes for r1
        genes_from_p1_to_r1 = self.parent_1[0:m + 1]
        genes_from_p2_to_r1 = self.parent_2[m + 1:]
        # collect genes for r2
        genes_from_p2_to_r2 = self.parent_2[0:m + 1]
        genes_from_p1_to_r2 = self.parent_1[m + 1:]
        # produce offspring
        r1 = genes_from_p1_to_r1 + genes_from_p2_to_r1
        r2 = genes_from_p2_to_r2 + genes_from_p1_to_r2
        return r1, r2

    def two_point_crossover(self):
        # avoid clones
        number_of_crossover_points = 2
        cross_points = random.sample(range(1, self.chromosome_length - 2), number_of_crossover_points)
        cross_points = np.sort(cross_points)
        # two-point crossover operator
        m1 = cross_points[0]
        m2 = cross_points[1]
        r1 = self.parent_1[0:m1 + 1] + self.parent_2[m1 + 1:m2 + 1] + self.parent_1[m2 + 1:]
        r2 = self.parent_2[0:m1 + 1] + self.parent_1[m1 + 1:m2 + 1] + self.parent_2[m2 + 1:]
        return r1, r2

    def three_point_crossover(self):
        # avoid clones
        number_of_crossover_points = 3
        cross_points = random.sample(range(1, self.chromosome_length - 2), number_of_crossover_points)
        cross_points = np.sort(cross_points)
        # three-point crossover operator
        m1 = cross_points[0]
        m2 = cross_points[1]
        m3 = cross_points[2]
        r1 = self.parent_1[0:m1 + 1] + self.parent_2[m1 + 1:m2 + 1] + self.parent_1[m2 + 1:m3 + 1] + self.parent_2[
                                                                                                     m3 + 1:]
        r2 = self.parent_2[0:m1 + 1] + self.parent_1[m1 + 1:m2 + 1] + self.parent_2[m2 + 1:m3 + 1] + self.parent_1[
                                                                                                     m3 + 1:]
        return r1, r2

    def uniform_crossover(self):
        r1 = []
        r2 = []
        for c in range(len(self.parent_1)):
            ar1 = np.zeros(self.chromosome_length)
            ar2 = np.zeros(self.chromosome_length)
            for i in range(self.chromosome_length):
                r = np.random.uniform(0.0, 1.0)
                if r >= 0.5:
                    ar1[i] = self.parent_1[c][i]
                    ar2[i] = self.parent_2[c][i]
                else:
                    ar2[i] = self.parent_1[c][i]
                    ar1[i] = self.parent_2[c][i]
            r1.append(ar1)
            r2.append(ar2)
        return r1, r2

    def uniform_crossover_1D(self):
        r1 = []
        r2 = []
        for i in range(self.chromosome_length):
            r = np.random.uniform(0.0, 1.0)
            if r >= 0.5:
                r1.append(self.parent_1[i])
                r2.append(self.parent_2[i])
            else:
                r1.append(self.parent_2[i])
                r2.append(self.parent_1[i])
        return r1, r2

    def commonality_based_crossover_1D(self):
        # chromosome length
        lc = self.chromosome_length
        # subset size of the first parent
        n1 = np.sum(self.parent_1)
        # subset size of the second parent
        n2 = np.sum(self.parent_2)
        # common bits
        common_bits = [i for i in range(lc) if self.parent_1[i] == self.parent_2[i]]
        # number of commonly selected features
        nc = sum([self.parent_1[i] for i in common_bits])
        # number of non-common bits
        nu = lc - len(common_bits)
        # initialize offspring
        o1 = list(np.zeros(lc, dtype=int))
        o2 = list(np.zeros(lc, dtype=int))
        # share common bits
        if len(common_bits) != 0:
            for i in common_bits:
                o1[i] = self.parent_1[i]
                o2[i] = self.parent_1[i]
        # calculate the probability for selecting the non-sharing features from parents
        r1 = (n1 - nc) / nu  # parent 1
        r2 = (n2 - nc) / nu  # parent 2
        # produce offspring
        for j in range(lc):
            if j not in common_bits:
                p = np.random.uniform(0.0, 1.0)
                if r1 >= r2:
                    if p > r2:
                        o1[j] = self.parent_1[j]
                        o2[j] = self.parent_2[j]
                    else:
                        o1[j] = self.parent_2[j]
                        o2[j] = self.parent_1[j]
                else:
                    if p > r1:
                        o1[j] = self.parent_2[j]
                        o2[j] = self.parent_1[j]
                    else:
                        o1[j] = self.parent_1[j]
                        o2[j] = self.parent_2[j]
        return o1, o2

    def commonality_based_crossover(self):
        # chromosome length
        lc = self.chromosome_length
        o1 = []
        o2 = []
        for c in range(len(self.parent_1)):
            p1 = self.parent_1[c]
            p2 = self.parent_2[c]
            # subset size of the first parent
            n1 = np.sum(p1)
            # subset size of the second parent
            n2 = np.sum(p2)
            # common bits
            common_bits = [i for i in range(lc) if p1[i] == p2[i]]
            # number of commonly selected features
            nc = sum([p1[i] for i in common_bits])
            # number of non-common bits
            nu = lc - len(common_bits)
            # initialize offspring
            ar1 = np.zeros(lc)
            ar2 = np.zeros(lc, dtype=int)
            # share common bits
            if len(common_bits) != 0:
                for i in common_bits:
                    ar1[i] = p1[i]
                    ar2[i] = p1[i]
            # calculate the probability for selecting the non-sharing features from parents
            r1 = (n1 - nc) / nu  # parent 1
            r2 = (n2 - nc) / nu  # parent 2
            # produce offspring
            for j in range(lc):
                if j not in common_bits:
                    p = np.random.uniform(0.0, 1.0)
                    if r1 >= r2:
                        if p > r2:
                            ar1[j] = p1[j]
                            ar2[j] = p2[j]
                        else:
                            ar1[j] = p2[j]
                            ar2[j] = p1[j]
                    else:
                        if p > r1:
                            ar1[j] = p2[j]
                            ar2[j] = p1[j]
                        else:
                            ar1[j] = p1[j]
                            ar2[j] = p2[j]
            o1.append(ar1)
            o2.append(ar2)
        return o1, o2
