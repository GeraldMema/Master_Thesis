class Evolutionary_Learning_Process:

    def __init__(self, parameters_selection, data_selection, methods_selection):
        self.params = parameters_selection
        self.data = data_selection
        self.methods = methods_selection

    def apply_evolutionary_clustering(self):
        # data transformation
        data_transformation = Data_Transformation(data_selection.dataset, \
                                                  methods_selection.evolutionary_learning_methods[
                                                      "data_transformation_method"])

        # initialize the population
        init_pop = \
            Population_Initialization(parameters_selection.evolutionary_learning_parameters["population_size"], \
                                      data_selection.features_set_size, \
                                      data_selection.features, \
                                      methods_selection.evolutionary_learning_methods[
                                          "population_initialization_method"])

        # evaluate initial population
        fitness_evaluation = Fitness_Evaluation(population=init_pop.initial_population,
                                                dataset=data_transformation.dataset)

        # evaluated population
        pop = fitness_evaluation.evaluated_population

        # store best solution
        selected_features_value = pop.sort_values("FITNESS_VALUE", ascending=False).loc[0, :]
        clustering_info_value = \
            fitness_evaluation.evaluation_details[pop.sort_values("FITNESS_VALUE", ascending=False).index.to_list()[0]]
        best_solution = {"Selected_Features": selected_features_value, "Clustering_Info": clustering_info_value}

        # sort population
        pop = pop.sort_values("FITNESS_VALUE", ascending=False)
        pop = pop.reset_index(drop=True)

        # array to host best fitness values
        fitness_values = []

        # store best fitness value
        value = best_solution["Selected_Features"].values[-1]
        fitness_values.append(value)

        # print the first solution
        number_of_clusters = best_solution["Clustering_Info"]["kmeans"]["kmeans_object"].n_clusters
        number_of_selected_features = len(best_solution["Clustering_Info"]["features"])
        print(
            f"Iteration 0: Cases have been grouped into {number_of_clusters} groups using {number_of_selected_features} features")
        print(f"Fitness value: {value}")

        # calculate the number of offspring
        crossover_percentage = parameters_selection.evolutionary_learning_parameters["crossover_percentage"]
        nc = int(2 * round(
            crossover_percentage * parameters_selection.evolutionary_learning_parameters["population_size"] / 2, 0))

        # calculate the number of mutants
        mutation_percentage = parameters_selection.evolutionary_learning_parameters["mutation_percentage"]
        nm = int(
            round(mutation_percentage * parameters_selection.evolutionary_learning_parameters["population_size"], 0))

        # GA main loop
        it = 1
        max_iterations = parameters_selection.evolutionary_learning_parameters["max_generations"]

        while it <= max_iterations:
            # while it <= 4:
            # recombination phase
            offspring = []
            # select parents
            parent_selection = Parent_Selection(pop, methods_selection.evolutionary_learning_methods[
                "parent_selection_method"])
            mate_selection = parent_selection.mate
            # apply crossover operator
            p1 = list(pop.loc[mate_selection[0], :].values[:-1].astype("int64"))
            p2 = list(pop.loc[mate_selection[1], :].values[:-1].astype("int64"))
            crossover = Crossover_Operator(p1, p2,
                                           methods_selection.evolutionary_learning_methods["crossover_operator"])
            o1 = crossover.offspring_1
            o2 = crossover.offspring_2
            offspring.append(o1)
            offspring.append(o2)
            k = 2
            while k < nc:
                # select parents
                parent_selection = Parent_Selection(pop, methods_selection.evolutionary_learning_methods[
                    "parent_selection_method"])
                mate_selection = parent_selection.mate
                # apply crossover operator
                p1 = list(pop.loc[mate_selection[0], :].values[:-1].astype("int64"))
                p2 = list(pop.loc[mate_selection[1], :].values[:-1].astype("int64"))
                crossover = Crossover_Operator(p1, p2,
                                               methods_selection.evolutionary_learning_methods["crossover_operator"])
                o1 = crossover.offspring_1
                o2 = crossover.offspring_2
                # check if new offspring has already been produced
                # while ((o1 in offspring) or (o2 in offspring)):
                #    p1 = list(pop.loc[mate_selection[0],:].values[:-1].astype("int64"))
                #    p2 = list(pop.loc[mate_selection[1],:].values[:-1].astype("int64"))
                #    crossover = Crossover_Operator(p1, p2)
                #    o1 = crossover.offspring_1
                #    o2 = crossover.offspring_2
                # continue producing
                offspring.append(o1)
                offspring.append(o2)
                k += 2
            pop_c = pd.DataFrame(data=offspring, columns=data_selection.features)

            # evaluate offspring
            crossover_fitness_evaluation = Fitness_Evaluation(population=pop_c, dataset=data_transformation.dataset)
            pop_crossover = crossover_fitness_evaluation.evaluated_population

            # mutation phase
            m = 1
            mutants = []
            while m <= nm:
                mutation = Mutation_Operator(pop,
                                             parameters_selection.evolutionary_learning_parameters["mutation_rate"], \
                                             methods_selection.evolutionary_learning_methods["mutation_operator"])
                if m > 1:
                    if mutation.mutant not in mutants:
                        mutants.append(mutation.mutant)
                        m += 1
                else:
                    mutants.append(mutation.mutant)
                    m += 1
            pop_m = pd.DataFrame(data=mutants, columns=data_selection.features)

            # evaluate mutants
            mutation_fitness_evaluation = Fitness_Evaluation(population=pop_m, dataset=data_transformation.dataset)
            pop_mutation = mutation_fitness_evaluation.evaluated_population

            # find best fitness value from offsprings
            best_crossover_solution = {
                "Selected_Features": pop_crossover.sort_values("FITNESS_VALUE", ascending=False).loc[0, :], \
                "Clustering_Info": crossover_fitness_evaluation.evaluation_details[
                    pop_crossover.sort_values("FITNESS_VALUE", ascending=False).index.to_list()[0]]}
            crossover_value = best_crossover_solution["Selected_Features"].values[-1]

            # find best fitness value from mutants
            best_mutation_solution = {
                "Selected_Features": pop_mutation.sort_values("FITNESS_VALUE", ascending=False).loc[0, :], \
                "Clustering_Info": mutation_fitness_evaluation.evaluation_details[
                    pop_mutation.sort_values("FITNESS_VALUE", ascending=False).index.to_list()[0]]}
            mutation_value = best_mutation_solution["Selected_Features"].values[-1]

            # check best fitness values from each population
            if crossover_value > mutation_value:
                if crossover_value > value:
                    fitness_values.append(crossover_value)
                    value = crossover_value
                    best_solution["Selected_Features"] = best_crossover_solution["Selected_Features"]
                    best_solution["Clustering_Info"] = best_crossover_solution["Clustering_Info"]
                else:
                    fitness_values.append(value)
            else:
                if mutation_value > value:
                    fitness_values.append(mutation_value)
                    value = mutation_value
                    best_solution["Selected_Features"] = best_mutation_solution["Selected_Features"]
                    best_solution["Clustering_Info"] = best_mutation_solution["Clustering_Info"]
                else:
                    fitness_values.append(value)

            # print best solution
            number_of_clusters = best_solution["Clustering_Info"]["kmeans"]["kmeans_object"].n_clusters
            number_of_selected_features = len(best_solution["Clustering_Info"]["features"])
            print(
                f"Iteration {it}: Cases have been grouped into {number_of_clusters} groups using {number_of_selected_features} features")
            print(f"Fitness value: {value}")

            # merge populations
            pop = pop.append(pop_crossover, ignore_index=True)
            pop = pop.append(pop_mutation, ignore_index=True)

            # sort population
            pop = pop.sort_values("FITNESS_VALUE", ascending=False)
            pop = pop.reset_index(drop=True)

            # truncation
            pop = pop.iloc[0:parameters_selection.evolutionary_learning_parameters["population_size"], :]

            # proceed to the next generation
            it += 1
        # Report extraction
        report = Evolutionary_Clustering_Report(fitness_values, best_solution, data_selection, parameters_selection, \
                                                methods_selection)
        print(
            f"File '{report.report_file}' has been generated and is available in the following path: {report.report_file_path}.")
