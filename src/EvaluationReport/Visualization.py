import matplotlib.pyplot as plt


class Visualization:

    def __init__(self):
        self.fitness_generation_plot = None
        self.error_analysis = None

    def plot_best_score_per_generation(self, solution_per_generation, lambdas, run):
        fig, ax = plt.subplots(1, 1)  # create figure & 1 axis
        fig.set_figheight(5)
        fig.set_figwidth(10)
        f = []
        a_ens = []
        d_avg = []
        for sol_info in solution_per_generation:
            f.append(sol_info[1])
            d_avg.append(sol_info[2])
            a_ens.append(sol_info[3])
        ax.plot(f, label='accuracy avg')
        ax.plot(d_avg, label='diversity avg')
        ax.plot(a_ens, label='accuracy ensemble')
        ax.plot(lambdas, label='lambda')
        ax.legend()
        ax.legend()
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        fig.savefig("Run" + str(run) + " ALL Scores per generation.png", bbox_inches='tight')
        plt.close('all')

    def plot_lambdas(self, lambdas, run):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        plt.plot(lambdas)
        plt.ylabel('Lambda')
        plt.xlabel('Generations')
        # fig.savefig("Run " + str(run) + " Lambda per Generation.png", bbox_inches='tight')
        plt.close('all')

    def plot_scores(self, my_alg, my_alg_not_GA, rf, xgb, gb, dt):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        runs = [i + 1 for i in range(len(my_alg))]

        plt.scatter(runs, rf, label='Random Forest')
        plt.scatter(runs, xgb, label='XGBoost')
        plt.scatter(runs, gb, label='Gradient Boosting')
        plt.scatter(runs, dt, label='Decision Trees')
        plt.scatter(runs, my_alg, label='My Algorithm')
        plt.scatter(runs, my_alg_not_GA, label='My Algorithm Without FS')

        plt.ylabel('Scores')
        plt.xlabel('Runs')
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                  ncol=2, mode="expand", borderaxespad=0.)
        fig.savefig("Evaluation with All", bbox_inches='tight')
        plt.close('all')

    def plot_error_analysis(self, fitness_scores, final_predictions, params, param_name, dataset):
        pass
        # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        # # plt.scatter(params, fitness_scores, label='best fitness score')
        # plt.scatter(params, final_predictions, label='final prediction score')
        # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #           ncol=2, mode="expand", borderaxespad=0.)
        # plt.ylabel('Scores')
        # plt.xlabel(param_name)
        # fig.savefig("Dataset " + str(dataset) + " " + param_name + " value.png", bbox_inches='tight')
        # plt.close('all')

    def plot_error_analysis_all(self, results, params, param_name, dataset):
        pass
        # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        # my_alg = []
        # my_alg_without_fs = []
        # rf = []
        # xgb = []
        # gb = []
        # dt = []
        # for i in range(len(params)):
        #     my_alg.append(results[i]['MY_ALG'][0])
        #     my_alg_without_fs.append(results[i]['MY_ALG_WITHOUT_GA'][0])
        #     rf.append(results[i]['RF'][0])
        #     xgb.append(results[i]['XGBoost'][0])
        #     gb.append(results[i]['GBoost'][0])
        #     dt.append(results[i]['DT'][0])
        # if (param_name == 'population_size') or (param_name == 'max_depth'):
        #     plt.plot(params, my_alg, label='My Algorithm')
        #     plt.plot(params, my_alg_without_fs, label='My Algorithm Without FS')
        #     plt.plot(params, rf, label='Random Forest')
        #     plt.plot(params, xgb, label='XGBoost')
        #     plt.plot(params, gb, label='Gradient Boosting')
        #     plt.plot(params, dt, label='Decision Trees')
        # else:
        #     plt.plot(params, my_alg, label='My Algorithm')
        #     plt.plot(params, my_alg_without_fs, label='My Algorithm Without FS')
        #     plt.plot(params, rf, label='Random Forest')
        #     plt.plot(params, xgb, label='XGBoost')
        #     plt.plot(params, gb, label='Gradient Boosting')
        #     plt.plot(params, dt, label='Decision Trees')
        # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #           ncol=2, mode="expand", borderaxespad=0.)
        # plt.ylabel('Scores')
        # plt.xlabel(param_name)
        # fig.savefig("Dataset " + str(dataset) + "Comparison per " + param_name + " value.png", bbox_inches='tight')
        # plt.close('all')

    def plot_error_analysis_mean(self, results, params, param_name, run):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        my_alg = []
        my_alg_without_fs = []
        rf = []
        xgb = []
        gb = []
        dt = []
        for i in range(len(params)):
            my_alg.append(results[i]['MY_ALG'][0])
            my_alg_without_fs.append(results[i]['MY_ALG_WITHOUT_GA'][0])
            rf.append(results[i]['RF'][0])
            xgb.append(results[i]['XGBoost'][0])
            gb.append(results[i]['GBoost'][0])
            dt.append(results[i]['DT'][0])
        if (param_name == 'population_size') or (param_name == 'max_depth'):
            plt.plot(params, my_alg, label='My Algorithm')
            plt.plot(params, my_alg_without_fs, label='My Algorithm Without FS')
            plt.plot(params, rf, label='Random Forest')
            plt.plot(params, xgb, label='XGBoost')
            plt.plot(params, gb, label='Gradient Boosting')
            plt.plot(params, dt, label='Decision Trees')
        else:
            plt.plot(params, my_alg, label='My Algorithm')
            plt.plot(params, my_alg_without_fs, label='My Algorithm Without FS')
            plt.plot(params, rf, label='Random Forest')
            plt.plot(params, xgb, label='XGBoost')
            plt.plot(params, gb, label='Gradient Boosting')
            plt.plot(params, dt, label='Decision Trees')
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                  ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Scores')
        plt.xlabel(param_name)
        fig.savefig("Run " + str(run) + "Comparison per " + param_name + " value.png", bbox_inches='tight')
        plt.close('all')

    def plot_ensembe_accuracy(self, solution_per_generation, run):
        fig, ax = plt.subplots(1, 1)  # create figure & 1 axis
        fig.set_figheight(5)
        fig.set_figwidth(10)
        fit_score = []
        a_ens = []
        for sol_info in solution_per_generation:
            fit_score.append(sol_info[0])
            a_ens.append(sol_info[3])
        ax.plot(fit_score, label='Fitness')
        ax.plot(a_ens, label='accuracy ensemble')
        ax.legend()
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        # fig.savefig("Run" + str(run) + " ENSEMBLE SCORE.png", bbox_inches='tight')
        # plt.close('all')

    def plot_ensembe_stats(self, ensemble_score_stats, run):

        fig, ax = plt.subplots()  # create figure & 1 axis
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        fig7, ax7 = plt.subplots()
        # fig.set_figheight(5)
        # fig.set_figwidth(10)
        a_ens = []
        fit_avg = []
        a_ens_gen = []
        a_avg_all = []
        a_avg = []
        a_std = []
        d_avg = []
        d_std = []
        for ensemble_stats in ensemble_score_stats:
            a_ens.append(ensemble_stats[0])
            fit_avg.append(ensemble_stats[1])
            a_ens_gen.append(ensemble_stats[6] + ensemble_stats[4])
            a_avg_all.append(ensemble_stats[4] + ensemble_stats[6] * (ensemble_stats[4] - ensemble_stats[6]))
            a_avg.append(ensemble_stats[4])
            a_std.append(ensemble_stats[5])
            d_avg.append(ensemble_stats[6])
            d_std.append(ensemble_stats[7])
        ax.scatter(a_ens, fit_avg, s=50, label='accuracy ensemble per fitness')
        ax.legend()
        plt.ylabel('fit_avg')
        plt.xlabel('a_ens')

        ax2.scatter(a_ens, a_ens_gen, s=50, label='accuracy ensemble per accuracy+ diversity')
        ax2.legend()
        plt.ylabel('a_ens_gen')
        plt.xlabel('a_ens')

        ax3.scatter(a_ens, a_avg_all, s=50, label='accuracy ensemble per per accuracy - diversity')
        ax3.legend()
        plt.ylabel('a_avg_all')
        plt.xlabel('a_ens')

        ax4.scatter(a_ens, a_avg, s=50, label='accuracy ensemble per accuracy')
        ax4.legend()
        plt.ylabel('a_avg')
        plt.xlabel('a_ens')

        ax5.scatter(a_ens, a_std, s=50, label='accuracy ensemble per accuracy std')
        ax5.legend()
        plt.ylabel('a_std')
        plt.xlabel('a_ens')

        ax6.scatter(a_ens, d_avg, s=50, label='accuracy ensemble per diversity accuracy')
        ax6.legend()
        plt.ylabel('d_avg')
        plt.xlabel('a_ens')

        ax7.scatter(a_ens, d_std, s=50, label='accuracy ensemble per diversity std')
        ax7.legend()
        plt.ylabel('d_std')
        plt.xlabel('a_ens')
        #
        # import os
        # print(os.getcwd())
        #
        # fig.savefig("Run" + str(run) + " ENSEMBLE SCORE STATS1.png", bbox_inches='tight')
        # fig2.savefig("Run" + str(run) + " ENSEMBLE SCORE STATS2.png", bbox_inches='tight')
        # fig3.savefig("Run" + str(run) + " ENSEMBLE SCORE STATS3.png", bbox_inches='tight')
        # fig4.savefig("Run" + str(run) + " ENSEMBLE SCORE STATS4.png", bbox_inches='tight')
        # fig5.savefig("Run" + str(run) + " ENSEMBLE SCORE STATS5.png", bbox_inches='tight')
        # fig6.savefig("Run" + str(run) + " ENSEMBLE SCORE STATS6.png", bbox_inches='tight')
        # fig7.savefig("Run" + str(run) + " ENSEMBLE SCORE STATS7.png", bbox_inches='tight')
        # plt.close('all')
