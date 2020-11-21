import operator
import numpy as np
import logging
import time

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
    solution_info_dict[i] = solution_info


class Evolutionary_Classification:

    def __init__(self, cfg):
        self.ens_predictions = None
        self.ens_score = -1
        self.previous_D = -1
        self.previous_A_ens = -1
        self.previous_A_avg = -1
        self.data_params = cfg['data_params']
        self.evolutionary_learning_params = cfg['evolutionary_learning_params']
        self.evolutionary_learning_methods = cfg['evolutionary_learning_methods']
        self.multiple_classification_models_params = cfg['multiple_classification_models_params']
        self.dynamic_ensemble_selection_params = cfg['dynamic_ensemble_selection_params']
        self.data_params = cfg['data_params']
        self.evaluation_params = cfg['evaluation_params']
        self.alpha = self.evolutionary_learning_params['fitness_lambda']

    def train_evaluate(self, cd, classifiers, data, features_per_classifiers, start_solution_idx, population_producer):
        mc = Multiple_Classifiers(self.multiple_classification_models_params, classifiers)
        for solution_idx in cd.solution_dict:
            clf_idx = solution_idx - start_solution_idx
            # fit with the corresponding training data
            X_train = cd.train_data_per_solution[solution_idx][clf_idx]
            y_train = data.y_train
            clf_model = mc.fit(X_train, y_train, clf_idx)
            # Predict from cross validation data
            X_cv = cd.cv_data_per_solution[solution_idx][clf_idx]
            mc.predict_per_classifier(X_cv, clf_model, clf_idx)
            # get scores from cross validation data
            y_cv = data.y_cv
            mc.get_score_per_classifier(y_cv, clf_idx)
            # keep the features per classifiers for future analysis
            features_per_classifiers[clf_idx] = X_train.columns
        # Predict  with the corresponding data for ensemble
        if population_producer == 'current':
            mc.predict_ensemble(len(data.y_cv))
            mc.get_score_ensemble(data.y_cv)
            self.ens_score = mc.score_ens
            self.ens_predictions = mc.predictions_ens
        else:  # WE DON'T CALCULATE THE ENSEMBLE SCORE FOR CROSSOVER/MUTATION POPULATION
            mc.score_ens = self.ens_score
            mc.predictions_ens = self.ens_predictions

        f = Fitness_Evaluation(self.evolutionary_learning_params, mc, self.evolutionary_learning_methods, data.y_cv,
                               self.alpha)
        return f

    def get_fitness_per_solution(self, data, population, population_producer, solution_idx, classifiers):
        # Get the corresponding data for each solution
        cd = Classifiers_Data(data)
        cd.extract_data_per_solution(population_producer, solution_idx, population)

        # For each solution get the fitness values
        fitness_values = {}
        solution_info_dict = {}
        features_per_classifiers = {}
        f = self.train_evaluate(cd, classifiers, data, features_per_classifiers, solution_idx, population_producer)
        for i in cd.solution_dict:
            fitness_values[i] = f.fitness_value[i - solution_idx]
            get_solution_info(solution_info_dict, cd, population_producer, features_per_classifiers, i, f, solution_idx)

        return fitness_values, solution_info_dict, f

    def apply_evolutionary_classification(self, cfg, run):

        # Get the classifiers
        c = Classifiers(cfg['multiple_classification_models_params'], cfg['evolutionary_learning_params'])
        # Get the data
        logging.info("Get and Process the Data")
        data = Data(cfg['data_params'], run)
        # TODO --> split random state
        data.process()

        # Measure the proposed algorithm performance
        start_time = time.time()

        # Initialize the population
        logging.info("Initialize the Population")
        population = Population(self.evolutionary_learning_params, self.evolutionary_learning_methods,
                                len(data.features), c.no_classifiers)
        population.random_initialization()

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
        best_solution_per_generation = []
        best_score = 0
        best_solution = None
        fitness_per_generation = {}

        while it <= max_iterations:
            logging.info("Generation no: " + str(it))
            # Get the fitness values from each current solution
            status = 'current'
            fitness_values, solution_info_dict, f_current = \
                self.get_fitness_per_solution(data, population, status, 0, c)

            # produce new crossover population
            status = 'crossover'
            solution_idx = max(fitness_values, key=int) + 1
            population.generate_crossover_population(solution_info_dict, fitness_values, nc)
            # Get the fitness values from each crossover
            fitness_values_crossover, solution_info_dict_crossover, _ = \
                self.get_fitness_per_solution(data, population, status, solution_idx, c)

            # Produce new mutation population
            status = 'mutation'
            solution_idx = max(fitness_values_crossover, key=int) + 1
            population.generate_mutation_population(solution_info_dict, nm)
            # Get the fitness values from each crossover
            fitness_values_mutation, solution_info_dict_mutation, _ = \
                self.get_fitness_per_solution(data, population, status, solution_idx, c)

            # concat all dicts
            all_fitness = {**fitness_values, **fitness_values_crossover, **fitness_values_mutation}
            all_solutions_info = {**solution_info_dict, **solution_info_dict_crossover, **solution_info_dict_mutation}

            # sort by fitness values
            sorted_fitness = dict(sorted(all_fitness.items(), key=operator.itemgetter(1), reverse=True))

            # find the best solution
            best_solution_position = list(sorted_fitness.keys())[0]
            best_current_score = list(sorted_fitness.values())[0]
            if best_current_score > best_score:
                best_score = best_current_score
                best_solution = all_solutions_info[best_solution_position]
            best_solution_per_generation.append(best_solution)
            logging.info('Best Fitness Score: ' + str(best_score))

            # Update the population
            n_best_solutions = {k: sorted_fitness[k] for k in list(sorted_fitness)[:population.max_population_size]}
            new_pop = []
            for i in n_best_solutions:
                # update population
                new_pop.append(all_solutions_info[i].chromosome)

            population.current_pop = np.unique(np.stack(new_pop, axis=0), axis=0)

            # adjust lambda
            lambdas.append(self.alpha)
            fitness_per_generation[it] = f_current
            if it > 1:
                self.previous_D = fitness_per_generation[it - 1].D_avg
                self.previous_A_ens = fitness_per_generation[it - 1].A_ens
                self.previous_A_avg = fitness_per_generation[it - 1].A_avg
                self.alpha = f_current.adjust_lambda(self.previous_A_ens, self.previous_A_avg, self.previous_D,
                                                     self.alpha)
            # proceed to the next generation
            it += 1

        stop = time.time() - start_time

        # Get the corresponding data from the best solution
        data_per_classifier = Classifiers_Data(data)
        train_data_per_classifier, test_data_per_classifier = \
            data_per_classifier.extract_data_for_ensemble(population.current_pop)

        # Get the evaluation results for my model
        evaluation_results = {}
        me = Models_Evaluation(self.evaluation_params)
        c.classifier_selection()
        mc = Multiple_Classifiers(self.multiple_classification_models_params, c)
        evaluation_results['MY_ALG'] = \
            me.my_alg_evalution(train_data_per_classifier, test_data_per_classifier, data.y_train, data.y_test, mc)
        # Get the evaluation results for the comparison models
        no_estimators = len(train_data_per_classifier)
        c.set_comparison_classifiers(no_estimators, run)
        for comparison_clf in c.comparison_classifiers:
            evaluation_results[comparison_clf] = me.other_evaluation(c.comparison_classifiers[comparison_clf],
                                                                     data.X_train, data.y_train, data.X_test,
                                                                     data.y_test)
        # plot the results
        plt_fitness = Visualization()
        plt_fitness.plot_best_score_per_generation(best_solution_per_generation)
        plt_fitness.plot_lambdas(lambdas)

        # report results
        report = Report(evaluation_results, best_solution, stop, c)

        return report, evaluation_results
