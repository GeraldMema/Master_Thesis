from src.DataProcessing.Data import Data
from src.MultipleClassificationModels.Multiple_Classifiers import Multiple_Classifiers
from src.EvolutionaryLearning.Solution_Info import Solution_Info
from src.EvolutionaryLearning.Solution_Representation import Solution_Representation
from src.EvolutionaryLearning.Population import Population
from src.MultipleClassificationModels.Classifiers_Data import Classifiers_Data
from src.EvolutionaryLearning.Fitness_Evaluation import Fitness_Evaluation


class Evolutionary_Classification:

    def __init__(self, cfg):
        self.data_params = cfg['data_params']
        self.evolutionary_learning_params = cfg['evolutionary_learning_params']
        self.evolutionary_learning_methods = cfg['evolutionary_learning_methods']
        self.multiple_classification_models_params = cfg['multiple_classification_models_params']
        self.dynamic_ensemble_selection_params = cfg['dynamic_ensemble_selection_params']
        self.data_params = cfg['data_params']
        self.evaluation_params = cfg['evaluation_params']

    def get_fitness_per_solution(self, data, population, multiple_classifiers, status):
        # Get the corresponding data for each solution
        classifiers_data = Classifiers_Data(data, population)
        classifiers_data.extract_data_per_solution(status)

        # For each solution get the fitness values
        fitness_values = {}
        for i in classifiers_data.solution_dict:
            # for each classifier
            for clf in classifiers_data.train_data_per_solution[i]:
                # fit with the corresponding training data
                X_train = classifiers_data.train_data_per_solution[i][clf]
                y_train = data.y_train
                clf_model = multiple_classifiers.fit(X_train, y_train, clf)
                # get scores from k fold cross validation
                multiple_classifiers.score(X_train, y_train, clf_model, clf)
                # Predict with the corresponding data for each classifier
                X_cv = classifiers_data.cv_data_per_solution[i][clf]
                multiple_classifiers.predict(X_cv, clf_model, clf)
            # Predict  with the corresponding data for ensemble
            multiple_classifiers.predict_ensemble(len(data.y_cv))

            f = Fitness_Evaluation(self.evolutionary_learning_params, multiple_classifiers)
            f.fitness_function()
            fitness_values[i] = f.fitness_value  # For each solution save the fitness value

        return classifiers_data.solution_dict, fitness_values

    def apply_evolutionary_clustering(self):

        # Get the data
        data = Data(self.data_params)
        data.process()
        # Get the features
        features = data.X_train.columns

        # Get the classifiers
        multiple_classifiers = Multiple_Classifiers(self.multiple_classification_models_params)
        # Get Classifiers as a list
        classifiers = list(multiple_classifiers.classifiers.keys())

        # Initialize the population
        solution_info = Solution_Info(classifiers, features)
        solution_representation = Solution_Representation(solution_info, self.evolutionary_learning_methods)
        population = Population(self.evolutionary_learning_params, self.evolutionary_learning_methods, solution_representation)
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
        while it <= max_iterations:

            # Get the fitness values from each current solution
            status = 'current'
            solution_dict, fitness_values = self.get_fitness_per_solution(data, population, multiple_classifiers, status)
            print('fitness: ', fitness_values)

            # Produce new crossover population
            status = 'crossover'
            solution_idx = max(solution_dict, key=int)
            population.generate_crossover_population(solution_dict, fitness_values, solution_idx, nc)
            # Get the fitness values from each crossover
            solution_dict_crossover, fitness_values_crossover = \
                self.get_fitness_per_solution(data, population, multiple_classifiers, status)

            # Produce new mutation population
            status = 'mutation'




            # Start Parent Selection, Crossover and Mutation methods (Recombination Phase)
            offsprings = []

            # proceed to the next generation
            it += 1

        # initialize the population
        # init_pop = \
        #     Population_Initialization(parameters_selection.evolutionary_learning_parameters["population_size"], \
        #                               data_selection.features_set_size, \
        #                               data_selection.features, \
        #                               methods_selection.evolutionary_learning_methods[
        #                                   "population_initialization_method"])
        #
        # # evaluate initial population
        # fitness_evaluation = Fitness_Evaluation(population=init_pop.initial_population,
        #                                         dataset=data_transformation.dataset)
        #
        # # evaluated population
        # pop = fitness_evaluation.evaluated_population
        #
        # # store best solution
        # selected_features_value = pop.sort_values("FITNESS_VALUE", ascending=False).loc[0, :]
        # clustering_info_value = \
        #     fitness_evaluation.evaluation_details[pop.sort_values("FITNESS_VALUE", ascending=False).index.to_list()[0]]
        # best_solution = {"Selected_Features": selected_features_value, "Clustering_Info": clustering_info_value}
        #
        # # sort population
        # pop = pop.sort_values("FITNESS_VALUE", ascending=False)
        # pop = pop.reset_index(drop=True)
        #
        # # array to host best fitness values
        # fitness_values = []
        #
        # # store best fitness value
        # value = best_solution["Selected_Features"].values[-1]
        # fitness_values.append(value)
        #
        # # print the first solution
        # number_of_clusters = best_solution["Clustering_Info"]["kmeans"]["kmeans_object"].n_clusters
        # number_of_selected_features = len(best_solution["Clustering_Info"]["features"])
        # print(
        #     f"Iteration 0: Cases have been grouped into {number_of_clusters} groups using {number_of_selected_features} features")
        # print(f"Fitness value: {value}")
        #
        # # calculate the number of offspring
        # crossover_percentage = parameters_selection.evolutionary_learning_parameters["crossover_percentage"]
        # nc = int(2 * round(
        #     crossover_percentage * parameters_selection.evolutionary_learning_parameters["population_size"] / 2, 0))
        #
        # # calculate the number of mutants
        # mutation_percentage = parameters_selection.evolutionary_learning_parameters["mutation_percentage"]
        # nm = int(
        #     round(mutation_percentage * parameters_selection.evolutionary_learning_parameters["population_size"], 0))
        #
        # # GA main loop
        # it = 1
        # max_iterations = parameters_selection.evolutionary_learning_parameters["max_generations"]
        #
        # while it <= max_iterations:
        #     # while it <= 4:
        #     # recombination phase
        #     offspring = []
        #     # select parents
        #     parent_selection = Parent_Selection(pop, methods_selection.evolutionary_learning_methods[
        #         "parent_selection_method"])
        #     mate_selection = parent_selection.mate
        #     # apply crossover operator
        #     p1 = list(pop.loc[mate_selection[0], :].values[:-1].astype("int64"))
        #     p2 = list(pop.loc[mate_selection[1], :].values[:-1].astype("int64"))
        #     crossover = Crossover_Operator(p1, p2,
        #                                    methods_selection.evolutionary_learning_methods["crossover_operator"])
        #     o1 = crossover.offspring_1
        #     o2 = crossover.offspring_2
        #     offspring.append(o1)
        #     offspring.append(o2)
        #     k = 2
        #     while k < nc:
        #         # select parents
        #         parent_selection = Parent_Selection(pop, methods_selection.evolutionary_learning_methods[
        #             "parent_selection_method"])
        #         mate_selection = parent_selection.mate
        #         # apply crossover operator
        #         p1 = list(pop.loc[mate_selection[0], :].values[:-1].astype("int64"))
        #         p2 = list(pop.loc[mate_selection[1], :].values[:-1].astype("int64"))
        #         crossover = Crossover_Operator(p1, p2,
        #                                        methods_selection.evolutionary_learning_methods["crossover_operator"])
        #         o1 = crossover.offspring_1
        #         o2 = crossover.offspring_2
        #         # check if new offspring has already been produced
        #         # while ((o1 in offspring) or (o2 in offspring)):
        #         #    p1 = list(pop.loc[mate_selection[0],:].values[:-1].astype("int64"))
        #         #    p2 = list(pop.loc[mate_selection[1],:].values[:-1].astype("int64"))
        #         #    crossover = Crossover_Operator(p1, p2)
        #         #    o1 = crossover.offspring_1
        #         #    o2 = crossover.offspring_2
        #         # continue producing
        #         offspring.append(o1)
        #         offspring.append(o2)
        #         k += 2
        #     pop_c = pd.DataFrame(data=offspring, columns=data_selection.features)
        #
        #     # evaluate offspring
        #     crossover_fitness_evaluation = Fitness_Evaluation(population=pop_c, dataset=data_transformation.dataset)
        #     pop_crossover = crossover_fitness_evaluation.evaluated_population
        #
        #     # mutation phase
        #     m = 1
        #     mutants = []
        #     while m <= nm:
        #         mutation = Mutation_Operator(pop,
        #                                      parameters_selection.evolutionary_learning_parameters["mutation_rate"], \
        #                                      methods_selection.evolutionary_learning_methods["mutation_operator"])
        #         if m > 1:
        #             if mutation.mutant not in mutants:
        #                 mutants.append(mutation.mutant)
        #                 m += 1
        #         else:
        #             mutants.append(mutation.mutant)
        #             m += 1
        #     pop_m = pd.DataFrame(data=mutants, columns=data_selection.features)
        #
        #     # evaluate mutants
        #     mutation_fitness_evaluation = Fitness_Evaluation(population=pop_m, dataset=data_transformation.dataset)
        #     pop_mutation = mutation_fitness_evaluation.evaluated_population
        #
        #     # find best fitness value from offsprings
        #     best_crossover_solution = {
        #         "Selected_Features": pop_crossover.sort_values("FITNESS_VALUE", ascending=False).loc[0, :], \
        #         "Clustering_Info": crossover_fitness_evaluation.evaluation_details[
        #             pop_crossover.sort_values("FITNESS_VALUE", ascending=False).index.to_list()[0]]}
        #     crossover_value = best_crossover_solution["Selected_Features"].values[-1]
        #
        #     # find best fitness value from mutants
        #     best_mutation_solution = {
        #         "Selected_Features": pop_mutation.sort_values("FITNESS_VALUE", ascending=False).loc[0, :], \
        #         "Clustering_Info": mutation_fitness_evaluation.evaluation_details[
        #             pop_mutation.sort_values("FITNESS_VALUE", ascending=False).index.to_list()[0]]}
        #     mutation_value = best_mutation_solution["Selected_Features"].values[-1]
        #
        #     # check best fitness values from each population
        #     if crossover_value > mutation_value:
        #         if crossover_value > value:
        #             fitness_values.append(crossover_value)
        #             value = crossover_value
        #             best_solution["Selected_Features"] = best_crossover_solution["Selected_Features"]
        #             best_solution["Clustering_Info"] = best_crossover_solution["Clustering_Info"]
        #         else:
        #             fitness_values.append(value)
        #     else:
        #         if mutation_value > value:
        #             fitness_values.append(mutation_value)
        #             value = mutation_value
        #             best_solution["Selected_Features"] = best_mutation_solution["Selected_Features"]
        #             best_solution["Clustering_Info"] = best_mutation_solution["Clustering_Info"]
        #         else:
        #             fitness_values.append(value)
        #
        #     # print best solution
        #     number_of_clusters = best_solution["Clustering_Info"]["kmeans"]["kmeans_object"].n_clusters
        #     number_of_selected_features = len(best_solution["Clustering_Info"]["features"])
        #     print(
        #         f"Iteration {it}: Cases have been grouped into {number_of_clusters} groups using {number_of_selected_features} features")
        #     print(f"Fitness value: {value}")
        #
        #     # merge populations
        #     pop = pop.append(pop_crossover, ignore_index=True)
        #     pop = pop.append(pop_mutation, ignore_index=True)
        #
        #     # sort population
        #     pop = pop.sort_values("FITNESS_VALUE", ascending=False)
        #     pop = pop.reset_index(drop=True)
        #
        #     # truncation
        #     pop = pop.iloc[0:parameters_selection.evolutionary_learning_parameters["population_size"], :]
        #
        #     # proceed to the next generation
        #     it += 1
        # # Report extraction
        # report = Evolutionary_Clustering_Report(fitness_values, best_solution, data_selection, parameters_selection, \
        #                                         methods_selection)
        # print(
        #     f"File '{report.report_file}' has been generated and is available in the following path: {report.report_file_path}.")
