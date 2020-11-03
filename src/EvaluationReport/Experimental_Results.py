import matplotlib.pyplot as plt


class Experimental_Results:

    def __init__(self, solution_per_generation):
        self.solution_per_generation = solution_per_generation

    def plot_best_score_per_generation(self):
        results = []
        for sol_info in self.solution_per_generation:
            results.append(sol_info.fitness_score)

        plt.plot(results)
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        plt.show()
