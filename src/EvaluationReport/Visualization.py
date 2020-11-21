import matplotlib.pyplot as plt


class Visualization:

    def __init__(self):
        self.fitness_generation_plot = None
        self.error_analysis = None

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
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        fig.savefig("fitness per generation.png", bbox_inches='tight')
        plt.close('all')

    def plot_error_analysis(self, fitness_scores, final_predictions, params, param_name):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        plt.scatter(params, fitness_scores, label = 'best fitness score')
        plt.scatter(params, final_predictions, label = 'final prediction score')
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Scores')
        plt.xlabel(param_name)
        fig.savefig("fitness and accuracy per " + param_name + " value.png", bbox_inches='tight')
        plt.close('all')

    def plot_lambdas(self, lambdas):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        plt.plot(lambdas)
        plt.ylabel('Lambda')
        plt.xlabel('Generations')
        fig.savefig("Lambda per Generation.png", bbox_inches='tight')
        plt.close('all')

    def plot_scores(self, my_alg, rf, xgb, gb, dt):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        runs = [i+1 for i in range(len(my_alg))]
        plt.scatter(runs, my_alg, label = 'My Algorithm')
        plt.scatter(runs, rf, label = 'Random Forest')
        plt.scatter(runs, xgb, label='XGBoost')
        plt.scatter(runs, gb, label='Gradient Boosting')
        plt.scatter(runs, dt, label='Decision Trees')
        plt.ylabel('Scores')
        plt.xlabel('Runs')
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
        fig.savefig("Evaluation with All", bbox_inches='tight')
        plt.close('all')
