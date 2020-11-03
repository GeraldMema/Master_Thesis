import operator
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import logging
import time

from src.EvaluationReport.Experimental_Results import Experimental_Results
from src.EvaluationReport.Models_Evaluation import Models_Evaluation
from src.EvaluationReport.Report import Report
from src.DataProcessing.Data import Data
from src.MultipleClassificationModels.Classifiers import Classifiers
from src.MultipleClassificationModels.Multiple_Classifiers import Multiple_Classifiers
from src.EvolutionaryLearning.Solution_Representation import Solution_Representation
from src.EvolutionaryLearning.Population import Population
from src.EvolutionaryLearning.Solution_Info import Solution_Info
from src.MultipleClassificationModels.Classifiers_Data import Classifiers_Data
from src.EvolutionaryLearning.Fitness_Evaluation import Fitness_Evaluation


class Evolutionary_Classification:

    def __init__(self, cfg, ROOT_DIR):
        self.data_params = cfg['data_params']
        self.evolutionary_learning_params = cfg['evolutionary_learning_params']
        self.evolutionary_learning_methods = cfg['evolutionary_learning_methods']
        self.multiple_classification_models_params = cfg['multiple_classification_models_params']
        self.dynamic_ensemble_selection_params = cfg['dynamic_ensemble_selection_params']
        self.data_params = cfg['data_params']
        self.evaluation_params = cfg['evaluation_params']
        self.root_dir = ROOT_DIR

    def get_fitness_per_solution(self, data, population, population_producer, solution_idx, classifiers):
        # Get the corresponding data for each solution
        logging.info('Get the corresponding data(selected features) for each selected classifier')
        cd = Classifiers_Data(data, classifiers)
        cd.extract_data_per_solution(population_producer, solution_idx, population)

        # For each solution get the fitness values
        fitness_values = {}
        solution_info_dict = {}
        for i in cd.solution_dict:
            logging.info(str(population_producer) + '. Solution no ' + str(i))
            # keep the info from each solution
            solution_info = Solution_Info(cd.solution_dict[i], population_producer)
            features_per_classifiers = {}
            # for each classifier
            mc = Multiple_Classifiers(self.multiple_classification_models_params, classifiers)
            for clf in cd.train_data_per_solution[i]:
                # fit with the corresponding training data
                X_train = cd.train_data_per_solution[i][clf]
                y_train = data.y_train
                clf_model = mc.fit(X_train, y_train, clf)

                # Predict from cross validation data
                X_cv = cd.cv_data_per_solution[i][clf]
                mc.predict(X_cv, clf_model, clf)

                # get scores from cross validation data
                y_cv = data.y_cv
                mc.score(y_cv, clf)

                # keep the features per classifiers for future analysis
                features_per_classifiers[clf] = X_train.columns
            # Predict  with the corresponding data for ensemble
            mc.predict_ensemble(len(data.y_cv))

            f = Fitness_Evaluation(self.evolutionary_learning_params, mc)
            f.fitness_function()
            fitness_values[i] = f.fitness_value  # For each solution save the fitness value

            # Solution Info
            solution_info.features_per_classifiers = features_per_classifiers
            solution_info.fitness_score = f.fitness_value
            solution_info_dict[i] = solution_info

        return cd.solution_dict, fitness_values, solution_info_dict

    def apply_evolutionary_classification(self):

        # Get the data
        logging.info("Get and Process the Data")
        data = Data(self.data_params)
        data.process()
        # Get the features
        features = data.features

        # Get the classifiers
        c = Classifiers(self.multiple_classification_models_params)
        classifiers_names = c.selected_classifiers

        # Measure the proposed algorithm performance
        start_time = time.time()

        # Initialize the population
        logging.info("Initialize the Population")
        solution_representation = \
            Solution_Representation(self.evolutionary_learning_methods, len(features), len(classifiers_names))
        population = Population(self.evolutionary_learning_params, self.evolutionary_learning_methods,
                                solution_representation)
        population.random_initialization()

        # Calculate crossover and mutation params
        logging.info("Calculate crossover and mutation params")
        crossover_percentage = float(self.evolutionary_learning_params['crossover_percentage'])
        mutation_percentage = float(self.evolutionary_learning_params['mutation_percentage'])
        population_size = int(self.evolutionary_learning_params['population_size'])
        nc = int(round(crossover_percentage * population_size, 0))  # number of crossover
        nm = int(round(mutation_percentage * population_size, 0))  # number of mutants

        # Until a stopping criterion is reached
        logging.info("Start the Looping")
        max_iterations = self.evolutionary_learning_params['max_generations']
        it = 1

        # Params
        best_solution_per_generation = []
        best_score = 0
        best_solution = None

        solution_idx = population_size + 1
        while (it <= max_iterations) and (solution_idx >= population_size):
            logging.info("Generation no: " + str(it))
            # Get the fitness values from each current solution
            status = 'current'
            solution_dict, fitness_values, solution_info_dict = \
                self.get_fitness_per_solution(data, population, status, 0, c)

            # Produce new crossover population
            status = 'crossover'
            solution_idx = max(solution_dict, key=int) + 1
            population.generate_crossover_population(solution_dict, fitness_values, nc)
            # Get the fitness values from each crossover
            solution_dict_crossover, fitness_values_crossover, solution_info_dict_crossover = \
                self.get_fitness_per_solution(data, population, status, solution_idx, c)

            # Produce new mutation population
            status = 'mutation'
            solution_idx = max(solution_dict_crossover, key=int) + 1
            population.generate_mutation_population(solution_dict, nm)
            # Get the fitness values from each crossover
            solution_dict_mutation, fitness_values_mutation, solution_info_dict_mutation = \
                self.get_fitness_per_solution(data, population, status, solution_idx, c)

            # concat all dicts
            all_fitness = {**fitness_values, **fitness_values_crossover, **fitness_values_mutation}
            all_solutions = {**solution_info_dict, **solution_info_dict_crossover, **solution_info_dict_mutation}

            # sort by fitness values
            sorted_d = dict(sorted(all_fitness.items(), key=operator.itemgetter(1), reverse=True))
            best_solution_position = list(sorted_d.keys())[0]
            best_current_score = list(sorted_d.values())[0]
            logging.info("Find the best Solution")
            if best_current_score > best_score:
                best_score = best_current_score
                best_solution = all_solutions[best_solution_position]
            best_solution_per_generation.append(best_solution)

            # Update the population
            n_best_solutions = {k: sorted_d[k] for k in list(sorted_d)[:population.max_population_size]}
            new_pop = []
            for i in n_best_solutions:
                new_pop.append(all_solutions[i].chromosome)
            population.current_pop = np.unique(np.stack(new_pop, axis=0), axis=0)

            logging.info('Best Fitness Score: ' + str(best_score))

            # proceed to the next generation
            it += 1

        stop = time.time() - start_time

        # Get the corresponding data from the best solution
        data_per_classifier = Classifiers_Data(data, c)
        train_data_per_classifier, test_data_per_classifier = \
            data_per_classifier.extract_test_data_for_ensemble(best_solution.chromosome, solution_representation)

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

        report = Report(evaluation_results, best_solution, stop, c)
        report.write_results(self.root_dir)

        res = Experimental_Results(best_solution_per_generation)
        res.plot_best_score_per_generation()
