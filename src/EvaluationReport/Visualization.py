import matplotlib.pyplot as plt
import numpy as np

class Visualization:

    def __init__(self):
        self.fitness_generation_plot = None
        self.error_analysis = None

    def plot_best_score_per_generation(self, solution_per_generation):
        f = []
        a = []
        d = []
        for sol_info in solution_per_generation:
            f.append(sol_info.fitness_score)
            a.append(sol_info.accuracy_score)
            d.append(sol_info.diversity_score)
        plt.plot(f, label = 'fitness')
        plt.plot(a, label = 'accuracy')
        plt.plot(d, label = 'diversity')
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        plt.show()
        self.fitness_generation_plot = plt.figure()

    def plot_error_analysis(self, scores, params, param_name):
        fig = plt.figure()
        plt.plot(params, scores)
        plt.ylabel('Best Score')
        plt.xlabel(param_name)
        self.error_analysis = fig
