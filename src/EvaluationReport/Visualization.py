import matplotlib.pyplot as plt
import numpy as np

class Visualization:

    def __init__(self, path):
        self.fitness_generation_plot = None
        self.error_analysis = None
        self.path = path

    def plot_best_score_per_generation(self, solution_per_generation):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        f = []
        a = []
        d = []
        for sol_info in solution_per_generation:
            f.append(sol_info.fitness_score)
            a.append(sol_info.accuracy_score)
            d.append(sol_info.diversity_score)
        ax.plot(f, label = 'fitness')
        ax.plot(a, label = 'accuracy')
        ax.plot(d, label = 'diversity')
        lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        fig.savefig("fitness per generation.png", bbox_inches='tight')

    def plot_error_analysis(self, scores, params, param_name):
        fig = plt.figure()
        plt.plot(params, scores)
        plt.ylabel('Best Score')
        plt.xlabel(param_name)
        self.error_analysis = fig
