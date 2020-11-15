import operator
import numpy as np
import logging
import time

from src.EvaluationReport.Visualization import Visualization
from src.EvaluationReport.Models_Evaluation import Models_Evaluation
from src.EvaluationReport.Report import Report
from src.MultipleClassificationModels.Multiple_Classifiers import Multiple_Classifiers
from src.EvolutionaryLearning.Solution_Representation import Solution_Representation
from src.EvolutionaryLearning.Population import Population
from src.EvolutionaryLearning.Solution_Info import Solution_Info
from src.MultipleClassificationModels.Classifiers_Data import Classifiers_Data
from src.EvolutionaryLearning.Fitness_Evaluation import Fitness_Evaluation


def get_solution_info(solution_info_dict, cd, pop_origin, features_per_classifiers, i, f, solution_idx):
    # keep the info from each solution
    if i > solution_idx:
        pop_origin = solution_info_dict[i-1].population_producer
    solution_info = Solution_Info(cd.solution_dict[i], pop_origin)
    # Solution Info
    solution_info.features_per_classifiers = features_per_classifiers[i]
    solution_info.fitness_score = f.fitness_value[i]
    solution_info.diversity_score = f.diversity_value[i]
    solution_info.accuracy_score = f.accuracy_value[i]
    solution_info_dict[i] = solution_info


class Evolutionary_Classification:

    def __init__(self, cfg):
        self.data_params = cfg['data_params']
        self.evolutionary_learning_params = cfg['evolutionary_learning_params']
        self.evolutionary_learning_methods = cfg['evolutionary_learning_methods']
        self.multiple_classification_models_params = cfg['multiple_classification_models_params']
        self.dynamic_ensemble_selection_params = cfg['dynamic_ensemble_selection_params']
        self.data_params = cfg['data_params']
        self.evaluation_params = cfg['evaluation_params']

    def train_evaluate(self, cd, population_producer, classifiers, data, features_per_classifiers, rep):
        mc = Multiple_Classifiers(self.multiple_classification_models_params, classifiers)
        clf = classifiers.classifier_dict[0]
        for i in cd.solution_dict:
            logging.info(str(population_producer) + '. Solution no ' + str(i))
            if rep == '2D':
                clf = classifiers.classifier_dict[i]
            # fit with the corresponding training data
            X_train = cd.train_data_per_solution[i][clf]
            y_train = data.y_train
            clf_model = mc.fit(X_train, y_train, clf)
            # Predict from cross validation data
            X_cv = cd.cv_data_per_solution[i][clf]
            mc.predict_per_solution(X_cv, clf_model, i)
            # get scores from cross validation data
            y_cv = data.y_cv
            mc.score_per_solution(y_cv, i)
            # keep the features per classifiers for future analysis
            features_per_classifiers[i] = X_train.columns
        # Predict  with the corresponding data for ensemble
        mc.predict_ensemble(len(data.y_cv))
        mc.score_ensemble(data.y_cv)

        f = Fitness_Evaluation(self.evolutionary_learning_params, mc, self.evolutionary_learning_methods, data.y_cv)
        return f

    def my_algorithm(self, data, population, c, nc, nm, best_solution_per_generation, best_score, best_solution,
                     representation_method):
        # Get the fitness values from each current solution
        status = 'current'
        solution_dict, fitness_values, solution_info_dict = \
            self.get_fitness_per_solution(data, population, status, 0, c, representation_method)

        # Produce new crossover population
        status = 'crossover'
        solution_idx = max(solution_dict, key=int) + 1
        population.generate_crossover_population(solution_dict, fitness_values, nc)
        # Get the fitness values from each crossover
        solution_dict_crossover, fitness_values_crossover, solution_info_dict_crossover = \
            self.get_fitness_per_solution(data, population, status, solution_idx, c, representation_method)

        # Produce new mutation population
        status = 'mutation'
        solution_idx = max(solution_dict_crossover, key=int) + 1
        population.generate_mutation_population(solution_dict, nm)
        # Get the fitness values from each crossover
        solution_dict_mutation, fitness_values_mutation, solution_info_dict_mutation = \
            self.get_fitness_per_solution(data, population, status, solution_idx, c, representation_method)

        # concat all dicts
        all_fitness = {**fitness_values, **fitness_values_crossover, **fitness_values_mutation}
        all_solutions_info = {**solution_info_dict, **solution_info_dict_crossover, **solution_info_dict_mutation}

        # sort by fitness values
        sorted_fitness = dict(sorted(all_fitness.items(), key=operator.itemgetter(1), reverse=True))

        best_solution_position = list(sorted_fitness.keys())[0]
        best_current_score = list(sorted_fitness.values())[0]
        if best_current_score > best_score:
            best_score = best_current_score
            best_solution = all_solutions_info[best_solution_position]
        best_solution_per_generation.append(best_solution)

        # Update the population
        n_best_solutions = {k: sorted_fitness[k] for k in list(sorted_fitness)[:population.max_population_size]}
        new_pop = []
        for i in n_best_solutions:
            new_pop.append(all_solutions_info[i].chromosome)
        population.current_pop = np.unique(np.stack(new_pop, axis=0), axis=0)

        logging.info('Best Fitness Score: ' + str(best_score))
        return best_score, best_solution

    def get_fitness_per_solution(self, data, population, population_producer, solution_idx, classifiers,
                                 representation):
        # Get the corresponding data for each solution
        cd = Classifiers_Data(data, classifiers)
        cd.extract_data_per_solution(population_producer, solution_idx, population)

        # For each solution get the fitness values
        fitness_values = {}
        solution_info_dict = {}
        features_per_classifiers = {}
        if representation == '1D':
            f = self.train_evaluate(cd, population_producer, classifiers, data, features_per_classifiers, representation)
            for i in cd.solution_dict:
                fitness_values[i] = f.fitness_value[i]
                get_solution_info(solution_info_dict, cd, population_producer, features_per_classifiers, i, f, solution_idx)
        elif representation == '2D':
            for i in cd.solution_dict:
                logging.info(str(population_producer) + '. Solution no ' + str(i))
                # for each classifier
                f = self.train_evaluate(cd, population_producer, classifiers, data, features_per_classifiers, representation)
                fitness_values[i] = f.fitness_value  # For each solution save the fitness value
                # keep the info from each solution
                get_solution_info(solution_info_dict, cd, population_producer, features_per_classifiers, i, f, solution_idx)

        return cd.solution_dict, fitness_values, solution_info_dict

    def apply_evolutionary_classification(self, c, data, path):

        # Get the features and classifiers
        features_names = data.features
        classifiers_names = c.selected_classifiers

        # Get Solution Representation info
        solution_representation = \
            Solution_Representation(self.evolutionary_learning_methods, len(features_names), len(classifiers_names))

        # Measure the proposed algorithm performance
        start_time = time.time()

        # Initialize the population
        logging.info("Initialize the Population")
        population = Population(self.evolutionary_learning_params, self.evolutionary_learning_methods,
                                solution_representation)
        population.random_initialization()

        # Calculate crossover and mutation params
        crossover_percentage = float(self.evolutionary_learning_params['crossover_percentage'])
        mutation_percentage = float(self.evolutionary_learning_params['mutation_percentage'])
        population_size = int(self.evolutionary_learning_params['population_size'])
        nc = int(round(crossover_percentage * population_size, 0))  # number of crossover
        nm = int(round(mutation_percentage * population_size, 0))  # number of mutants

        # Until a stopping criterion is reached
        max_iterations = self.evolutionary_learning_params['max_generations']
        it = 1

        # Params
        best_solution_per_generation = []
        best_score = 0
        best_solution = None
        while it <= max_iterations:
            logging.info("Generation no: " + str(it))
            best_score, best_solution = self.my_algorithm(data, population, c, nc, nm,
                                                          best_solution_per_generation, best_score,
                                                          best_solution,
                                                          solution_representation.representation_method)
            # proceed to the next generation
            it += 1

        stop = time.time() - start_time

        # Get the corresponding data from the best solution
        data_per_classifier = Classifiers_Data(data, c)
        train_data_per_classifier, test_data_per_classifier = None, None
        if solution_representation.representation_method == '2D':
            train_data_per_classifier, test_data_per_classifier = \
                data_per_classifier.extract_data_for_ensemble_2D(best_solution.chromosome)
        elif solution_representation.representation_method == '1D':
            train_data_per_classifier, test_data_per_classifier = \
                data_per_classifier.extract_data_for_ensemble_1D(population.current_pop, c)

        # Get the evaluation results
        evaluation_results = {}
        me = Models_Evaluation(self.evaluation_params)

        evaluation_results['MY_ALG'] = \
            me.my_alg_evalution(train_data_per_classifier, test_data_per_classifier, data.y_train, data.y_test,
                                self.multiple_classification_models_params)

        for comparison_clf in c.comparison_classifiers:
            evaluation_results[comparison_clf] = me.other_evaluation(c.comparison_classifiers[comparison_clf],
                                                                     data.X_train, data.y_train, data.X_test,
                                                                     data.y_test)
        # plot the results
        plt_fitness = Visualization(path)
        plt_fitness.plot_best_score_per_generation(best_solution_per_generation)

        # report results
        report = Report(evaluation_results, best_solution, stop, c)

        return report
