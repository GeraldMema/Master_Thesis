import numpy as np
import random
import numpy.random as npr


def roulette_wheel_selection(population, fitness_values):
    # function to fortune wheel
    def fortune_wheel():
        max = sum([fitness_values[i] for i in fitness_values])
        selection_probs = [fitness_values[i] / max for i in fitness_values]
        return npr.choice(len(fitness_values), p=selection_probs)

    mate = []
    p1 = fortune_wheel()
    p2 = fortune_wheel()
    while p2 == p1:
        p1 = fortune_wheel()
        p2 = fortune_wheel()
    mate.append(population.current_pop.iloc[p1])
    mate.append(population.current_pop.iloc[p2])
    return tuple(mate)


def tournament_selection(population, fitness_values):
    # function to create a tournament
    def tournament():
        tournament_size = round(0.7 * len(fitness_values))
        random_sample = random.sample(range(0, len(fitness_values)), tournament_size)
        corresponding_fitness_values = [fitness_values[i] for i in random_sample]
        if len(corresponding_fitness_values) > 0:
            position = corresponding_fitness_values.index(max(corresponding_fitness_values))
        else:
            position = 0
        return random_sample[position]

    # select parents
    mate = []
    p1 = tournament()
    p2 = tournament()
    while p2 == p1:
        p1 = tournament()
        p2 = tournament()
    mate.append(population.current_pop.iloc[p1])
    mate.append(population.current_pop.iloc[p2])
    return tuple(mate)


def random_selection(population):
    mate = []
    p1 = np.random.choice(list(population.current_pop.index))
    p2 = np.random.choice(list(population.current_pop.index))

    i = 0
    while p2 == p1:
        p1 = np.random.choice(population.current_pop.index)
        p2 = np.random.choice(population.current_pop.index)
        if i == 10:
            p1 = 1
            p2 = 2
            break
        i += 1

    mate.append(population.current_pop.loc[p1])
    mate.append(population.current_pop.loc[p2])
    return tuple(mate)


class Parent_Selection:
    def __init__(self, population, fitness_values, evolutionary_learning_methods):
        self.parent_selection_method = evolutionary_learning_methods['parent_selection_method']
        if self.parent_selection_method == 'roulette_wheel_selection':
            self.mate = roulette_wheel_selection(population, fitness_values)
        elif self.parent_selection_method == 'tournament_selection':
            self.mate = tournament_selection(population, fitness_values)
        else:
            self.mate = random_selection(population)
