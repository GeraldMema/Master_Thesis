import operator
import numpy as np
import logging
import time
from sklearn import tree
import matplotlib.pyplot as plt

from src.EvaluationReport.Visualization import Visualization
from src.EvaluationReport.Models_Evaluation import Models_Evaluation
from src.EvaluationReport.Report import Report
from src.MultipleClassificationModels.Multiple_Classifiers import Multiple_Classifiers
from src.MultipleClassificationModels.Classifiers import Classifiers
from src.DataProcessing.Data import Data
from src.EvolutionaryLearning.Population import Population
from src.EvolutionaryLearning.Solution_Info import Solution_Info
from src.MultipleClassificationModels.Classifiers_Data import Classifiers_Data
from src.EvolutionaryLearning.Fitness_Evaluation import Fitness_Evaluation

import pandas as pd


def get_solution_info(solution_info_dict, cd, pop_origin, features_per_classifiers, i, f, solution_idx):
    # keep the info from each solution

    if i > solution_idx:
        pop_origin = solution_info_dict[i - 1].population_producer
    solution_info = Solution_Info(cd.solution_dict[i], pop_origin)
    # Solution Info

    solution_info.features_per_classifiers = features_per_classifiers[i - solution_idx]
    solution_info.fitness_score = f.fitness_value[i - solution_idx]
    solution_info.diversity_score = f.Di[i - solution_idx]
    solution_info.accuracy_score = f.Ai[i - solution_idx]
    solution_info.accuracy_ens_score = f.A_ens
    solution_info.accuracy_avg = f.A_avg
    solution_info.diversity_avg = f.D_avg
    solution_info.index = i
    solution_info_dict[i] = solution_info


def get_pop_prod(index, min_cross, max_cross, min_mut, max_mut):
    if (index >= min_cross) & (index <= max_cross):
        return 'CROSSOVER'
    elif (index >= min_mut) & (index <= max_mut):
        return 'MUTATION'
    else:
        return 'CURRENT'


def get_acc(idx, min_cross, max_cross, min_mut, max_mut, f_current, f_crossover, f_mutation):
    if (idx >= min_cross) & (idx <= max_cross):
        return f_crossover.Ai[idx]
    elif (idx >= min_mut) & (idx <= max_mut):
        return f_mutation.Ai[idx]
    else:
        return f_current.Ai[idx]


def get_div(idx, min_cross, max_cross, min_mut, max_mut, f_current, f_crossover, f_mutation):
    if (idx >= min_cross) & (idx <= max_cross):
        return f_crossover.Di[idx]
    elif (idx >= min_mut) & (idx <= max_mut):
        return f_mutation.Di[idx]
    else:
        return f_current.Di[idx]


class Evolutionary_Classification:

    def __init__(self, cfg):
        self.ens_predictions = None
        self.ens_score = 1
        self.ens_score_test = 1
        self.previous_D = 1
        self.previous_A_ens = 1
        self.previous_A_avg = 1
        self.data_params = cfg['data_params']
        self.evolutionary_learning_params = cfg['evolutionary_learning_params']
        self.evolutionary_learning_methods = cfg['evolutionary_learning_methods']
        self.multiple_classification_models_params = cfg['multiple_classification_models_params']
        self.dynamic_ensemble_selection_params = cfg['dynamic_ensemble_selection_params']
        self.data_params = cfg['data_params']
        self.evaluation_params = cfg['evaluation_params']
        self.alpha = self.evolutionary_learning_params['fitness_lambda']
        self.change_dataset = self.evolutionary_learning_methods['change_dataset']
        self.lf = self.evolutionary_learning_params['learning_factor']
        self.set_lambda_after = self.evolutionary_learning_params['set_lambda_after']

    def train_evaluate(self, cd, classifiers, data, population_producer):
        """
        Add description here
        """
        mc = Multiple_Classifiers(self.multiple_classification_models_params, classifiers)
        mc_test = Multiple_Classifiers(self.multiple_classification_models_params, classifiers)
        for solution_idx in cd.solution_dict:
            # fit with the corresponding training data
            X_train = cd.train_data_per_solution[solution_idx]
            y_train = data.y_train
            clf_model = mc.fit(X_train, y_train)
            # Predict from cross validation data
            X_cv = cd.cv_data_per_solution[solution_idx]
            X_test = cd.test_data_per_solution[solution_idx]
            mc.predict_per_classifier(X_cv, clf_model, solution_idx)
            mc_test.predict_per_classifier(X_test, clf_model, solution_idx)
            # get scores from cross validation data
            y_cv = data.y_cv
            y_test = data.y_test
            mc.get_score_per_classifier(y_cv, solution_idx)
            mc_test.get_score_per_classifier(y_test, solution_idx)
            # if population_producer == 'current':
            #
            #     print('SCORE FOR CLASSIFIER {} is {}'.format(solution_idx,mc.score_per_classifier[solution_idx]))
            #     print('Features look: {}'.format(X_train.columns))
            #     print('ENSEMBLE SCORE: {}'.format(self.ens_score))
            #     print('ENSEMBLE SCORE TEST: {}'.format(self.ens_score_test))
            #     print()
            #     if mc.score_per_classifier[solution_idx] == 1:
            #         plt.rcParams["figure.figsize"] = (60, 18)
            #         tree.plot_tree(clf_model)
            #         plt.show()
        # Predict  with the corresponding data for ensemble
        if population_producer == 'current':

            mc.predict_ensemble(len(data.y_cv), y_test=data.y_cv)
            mc.get_score_ensemble(data.y_cv)
            mc_test.predict_ensemble(len(data.y_test))
            mc_test.get_score_ensemble(data.y_test)
            self.ens_score = mc.score_ens
            self.ens_score_test = mc_test.score_ens
            self.ens_predictions = mc.predictions_ens
            # print('ENSEMBLE SCORE: {}'.format(self.ens_score))
            # print('ENSEMBLE SCORE TEST: {}'.format(self.ens_score_test))
        else:  # WE DON'T CALCULATE THE ENSEMBLE SCORE FOR CROSSOVER/MUTATION POPULATION
            mc.score_ens = self.ens_score
            mc.predictions_ens = self.ens_predictions

        f = Fitness_Evaluation(self.evolutionary_learning_params, mc, self.evolutionary_learning_methods, data.y_cv,
                               self.alpha)
        return f

    def get_fitness_per_solution(self, data, population, population_producer, classifiers):
        # Get the corresponding data for each solution
        cd = Classifiers_Data(data)
        cd.extract_data_per_solution(population_producer, population)

        # For each solution get the fitness values
        f = self.train_evaluate(cd, classifiers, data, population_producer)
        return f.fitness_value, f

    def get_fitness_per_solution_old(self, data, population, population_producer, solution_idx, classifiers):
        # Get the corresponding data for each solution
        cd = Classifiers_Data(data)
        cd.extract_data_per_solution(population_producer, population)

        # For each solution get the fitness values
        fitness_values = {}
        solution_info_dict = {}
        f = self.train_evaluate(cd, classifiers, data, population_producer)
        for i in cd.solution_dict:
            fitness_values[i] = f.fitness_value[i - solution_idx]
            get_solution_info(solution_info_dict, cd, population_producer, i, f, solution_idx)

        return fitness_values, solution_info_dict, f

    def apply_evolutionary_classification(self, cfg, run, classification_dataset):

        st = time.time()

        # Get the classifiers
        c = Classifiers(cfg['multiple_classification_models_params'], cfg['evolutionary_learning_params'], 10)
        c_1 = Classifiers(cfg['multiple_classification_models_params'], cfg['evolutionary_learning_params'], 20)
        c_without = Classifiers(cfg['multiple_classification_models_params'], cfg['evolutionary_learning_params'], 30)

        # Measure the proposed algorithm performance
        start_time = time.time()

        # Get the data
        logging.info("Get and Process the Data")
        data = Data(cfg['data_params'], 1)
        d, t = data.process(classification_dataset)
        if d is None:
            return None, None
        data.split_data(run, d, t)

        # Initialize the population
        logging.info("Initialize the Population")
        population = Population(self.evolutionary_learning_params, self.evolutionary_learning_methods,
                                len(data.features), c.no_classifiers)

        # Calculate crossover and mutation params
        crossover_percentage = float(self.evolutionary_learning_params['crossover_percentage'])
        mutation_percentage = float(self.evolutionary_learning_params['mutation_percentage'])
        population_size = int(self.evolutionary_learning_params['population_size'])
        nc = int(round(crossover_percentage * population_size, 0))  # number of crossover
        nm = int(round(mutation_percentage * population_size, 0))  # number of mutants

        # trade off param of accuracy/diversity
        lambdas = []

        # Until a stopping criterion is reached
        max_iterations = self.evolutionary_learning_params['max_generations']
        it = 1

        # Params
        behavior_per_generation = []
        best_score = 0
        best_solution = None
        ensemble_score_stats = []
        current_pops = []
        stats = []
        generation_with_higher_score = -1

        solutions_per_generations = []
        while it <= max_iterations:
            if self.set_lambda_after:
                if (max_iterations / 2) > it:
                    self.alpha = 0
                elif (max_iterations / 2) == it:
                    self.alpha = self.evolutionary_learning_params['fitness_lambda']

            # Get the fitness values from each current solution

            if self.change_dataset:
                if it % 8 == 0:
                    data.split_data(it, d, t)

            fitness_values, f_current = \
                self.get_fitness_per_solution(data, population, 'current', c)
            # print('END CURRENT')
            # produce new crossover population
            population.generate_crossover_population(population, fitness_values, nc)
            # Get the fitness values from each crossover
            fitness_values_crossover, f_crossover = \
                self.get_fitness_per_solution(data, population, 'crossover', c)
            # print('END CROSSOVER')
            # Produce new mutation population
            population.generate_mutation_population(population, nm)
            # Get the fitness values from each crossover
            fitness_values_mutation, f_mutation = \
                self.get_fitness_per_solution(data, population, 'mutation', c)
            # print('END MUTATION')
            # concat all dicts
            min_cross = min(list(fitness_values_crossover.keys()))
            max_cross = max(list(fitness_values_crossover.keys()))
            min_mut = min(list(fitness_values_mutation.keys()))
            max_mut = max(list(fitness_values_mutation.keys()))
            all_fitness = {**fitness_values, **fitness_values_crossover, **fitness_values_mutation}
            # sort by fitness values
            sorted_fitness = dict(sorted(all_fitness.items(), key=operator.itemgetter(1), reverse=True))

            # Update the population
            sorted_solutions = {k: sorted_fitness[k] for k in list(sorted_fitness)}
            sorted_solutions_features = {}
            for sol in sorted_solutions:
                if sol in population.current_pop.index:
                    sorted_solutions_features[sol] = population.current_pop.loc[sol]
                elif sol in population.crossover_pop.index:
                    sorted_solutions_features[sol] = population.crossover_pop.loc[sol]
                else:
                    sorted_solutions_features[sol] = population.mutation_pop.loc[sol]
            sorted_solutions_df = pd.DataFrame.from_dict(sorted_solutions_features, orient='index')
            population.current_pop = sorted_solutions_df.drop_duplicates().head(population_size)

            fitness_values, f_current = \
                self.get_fitness_per_solution(data, population, 'current', c)

            sols = population.current_pop.copy()

            # adjust lambda
            # all_A = {**f_current.Ai, **f_crossover.Ai, **f_mutation.Ai}
            all_A = {**f_current.Ai}
            A_avg = sum(all_A.values()) / len(all_A)
            A_gen = A_avg + f_current.D_avg
            stats.append((A_gen, A_avg, f_current.D_avg, f_current.A_ens))
            lambdas.append(self.alpha)
            if self.set_lambda_after:
                if (max_iterations / 2) <= it:
                    self.alpha = f_current.adjust_lambda(stats, self.alpha, self.ens_score, A_avg)
            else:
                self.alpha = f_current.adjust_lambda(stats, self.alpha, self.ens_score, A_avg)

            sols['Iteration'] = it
            sols['Population_Producer'] = sols.apply(
                lambda x: get_pop_prod(x.name, min_cross, max_cross, min_mut, max_mut), axis=1)
            sols['Fitness_Score'] = sols.apply(lambda x: all_fitness[x.name], axis=1)
            sols['Accuracy_Score'] = sols.apply(lambda x: f_current.Ai[x.name], axis=1)
            sols['Diversity_Score'] = sols.apply(lambda x: f_current.Di[x.name], axis=1)
            sols['Ensemble_Score'] = sols.apply(lambda x: f_current.A_ens, axis=1)
            sols['Ensemble_Score_test'] = sols.apply(lambda x: self.ens_score_test, axis=1)
            sols['Fitness_Avg_Score'] = sols.apply(lambda x: f_current.F_avg, axis=1)
            sols['Acc_Avg_Score'] = sols['Accuracy_Score'].mean()
            sols['Acc_Gen_Score'] = sols.apply(lambda x: A_gen, axis=1)
            sols['Div_Avg_Score'] = sols['Diversity_Score'].mean()
            sols['lambda'] = sols.apply(lambda x: self.alpha, axis=1)

            solutions_per_generations.append(sols)

            # # Have we achieved better results???
            # if len(stats) > 1:
            #     case_to_update = f_current.A_ens > (stats[len(stats) - 2][3] * self.lf)
            #     if not case_to_update:
            #         population.current_pop = current_pops[it - 1]

            # update best solution
            behavior_per_generation.append([f_current.F_avg, f_current.A_avg, f_current.D_avg, f_current.A_ens])
            logging.info('Ensemble Score: ' + str(self.ens_score))
            logging.info('ens_score_test: ' + str(self.ens_score_test))
            logging.info('Fitness Avg Score: ' + str(f_current.F_avg))
            logging.info('Ensemble generalization: ' + str(A_gen))
            logging.info('Accuracy avg all: ' + str(A_avg))
            logging.info('Accuracy avg: ' + str(f_current.A_avg))
            # logging.info('Accuracy std: ' + str(f_current.A_std))
            logging.info('Diversity avg: ' + str(f_current.D_avg))
            # logging.info('Diversity std: ' + str(f_current.D_std))
            logging.info('lambda: ' + str(self.alpha))

            logging.info("---------------------------------------------------------------------------------")
            ensemble_score_stats.append((self.ens_score, f_current.F_avg, A_gen, A_avg, f_current.A_avg,
                                         f_current.A_std, f_current.D_avg, f_current.D_std))

            # proceed to the next generation
            it += 1

        stop = time.time() - start_time

        # Get the corresponding data from the best solution
        # update the population current score with the best score
        logging.info('BEST SCORE IN GENERATION: {}'.format(generation_with_higher_score + 1))
        logging.info('BEST SCORE: {}'.format(best_score))

        final_data = data
        final_data.X_train = d
        final_data.y_train = t

        # self.show_features_per_classifier(population.current_pop)
        data_per_classifier = Classifiers_Data(final_data)
        train_data_per_classifier, test_data_per_classifier = \
            data_per_classifier.extract_data_for_ensemble(population.current_pop)
        data_per_classifier_w = Classifiers_Data(final_data)

        # Get the evaluation results for my model
        evaluation_results = {}
        me = Models_Evaluation(self.evaluation_params)
        mc = Multiple_Classifiers(self.multiple_classification_models_params, c)
        temp_results = \
            me.my_alg_evalution(train_data_per_classifier, test_data_per_classifier, final_data.y_train,
                                final_data.y_test, mc)

        evaluation_results['MY_ALG'] = (temp_results[0], round((time.time() - st), 2),temp_results[2],temp_results[3])

        max_runs = 10
        scores_without = []
        st2 = time.time()
        for _ in range(max_runs):
            mc_without = Multiple_Classifiers(self.multiple_classification_models_params, c_without)
            population_w = Population(self.evolutionary_learning_params, self.evolutionary_learning_methods,
                                      len(final_data.features), c_without.no_classifiers)
            # self.show_features_per_classifier(population_w.current_pop)
            train_data_per_classifier_without, test_data_per_classifier_without = \
                data_per_classifier_w.extract_data_for_ensemble(population_w.current_pop)
            score_without = me.my_alg_evalution(train_data_per_classifier_without, test_data_per_classifier_without,
                                                final_data.y_train,
                                                final_data.y_test, mc_without)
            scores_without.append(score_without[0])
        evaluation_results['MY_ALG_WITHOUT_GA'] = ((sum(scores_without) / len(scores_without)), (time.time()-st2)/max_runs)
        # Get the evaluation results for the comparison models
        no_estimators = len(train_data_per_classifier)
        c.set_comparison_classifiers(no_estimators, run)
        for comparison_clf in c.comparison_classifiers:
            evaluation_results[comparison_clf] = me.other_evaluation(c.comparison_classifiers[comparison_clf],
                                                                     final_data.X_train, final_data.y_train,
                                                                     final_data.X_test,
                                                                     final_data.y_test, comparison_clf == 'RF')
        # plot the results
        vis = Visualization()
        # vis.plot_ensembe_stats(ensemble_score_stats, run)
        vis.plot_best_score_per_generation(behavior_per_generation, lambdas, run)
        vis.plot_ensembe_accuracy(behavior_per_generation, run)

        final_report_df = pd.concat(solutions_per_generations)
        for alg in evaluation_results:
            final_report_df[alg] = evaluation_results[alg][0]
        final_report_df.to_csv(r'C:/Users/bayer/Documents/thesis/final_report.csv')
        # report results
        report = Report(evaluation_results, best_solution, stop, c)

        return report, evaluation_results

    def show_features_per_classifier(self, current_pop):
        l = list(current_pop)
        feat_per_class = {}
        for clf, a in enumerate(l):
            a_list = list(a)
            feat_per_class[clf] = [i + 1 for i, e in enumerate(a_list) if e == 1]
        for k in feat_per_class:
            logging.info(feat_per_class[k])
