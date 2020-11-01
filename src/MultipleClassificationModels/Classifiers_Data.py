class Classifiers_Data:
    """
    Add Description
    """

    def __init__(self, data, population):
        self.data = data
        self.pop = population
        self.train_data_per_solution = {}
        self.cv_data_per_solution = {}
        self.test_data_per_solution = {}
        self.solution_dict = {}

    def extract_data_per_solution(self, status):
        """
        This function will return a list of dicts with all the needed data info for each classifier. The
        list represent all the current solutions that we have in a population. The dict will contain the classifier name
        as key and a pandas dataframe with the subset of selected features.

        :param
        :return:
        """

        population_size = self.pop.current_pop.shape[0]
        classifiers_dict = self.pop.solution_representation.solution_info.classifiers_dict
        features_dict = self.pop.solution_representation.solution_info.features_dict

        for p in range(population_size):
            if status == 'crossover':
                solution = self.pop.crossover_pop[p]
            elif status == 'mutation':
                solution = self.pop.mutation_pop[p]
            else:
                solution = self.pop.current_pop[p]
            train_data_per_classifier = {}
            cv_data_per_classifier = {}
            test_data_per_classifier = {}
            if self.pop.solution_representation.representation_method == '1D':
                # features
                no_features = self.pop.solution_representation.solution_info.no_features
                selected_features = solution[:no_features]
                features = [features_dict[idx] for idx, i in enumerate(selected_features) if i == 1]
                # classifiers
                selected_classifiers = solution[no_features:]
                classifiers = [classifiers_dict[idx] for idx, i in enumerate(selected_classifiers) if i == 1]
                for clf in classifiers:
                    train_data_per_classifier[clf] = self.data.X_train[features]
                    cv_data_per_classifier[clf] = self.data.X_cv[features]
                    test_data_per_classifier[clf] = self.data.X_test[features]
            if self.pop.solution_representation.representation_method == '2D':
                for idx, selected_features in enumerate(solution):
                    clf = classifiers_dict[idx]
                    features = [features_dict[i] for i in selected_features if i == 1]
                    train_data_per_classifier[clf] = self.data.X_train[features]
                    cv_data_per_classifier[clf] = self.data.X_cv[features]
                    test_data_per_classifier[clf] = self.data.X_test[features]
            if self.pop.solution_representation.representation_method == 'dual':
                # TODO: apply this based on the paper
                continue

            # self.train_data_per_solution.append(train_data_per_classifier)
            # self.cv_data_per_solution.append(cv_data_per_classifier)
            # self.test_data_per_solution.append(test_data_per_classifier)

            self.train_data_per_solution[p] = train_data_per_classifier
            self.cv_data_per_solution[p] = cv_data_per_classifier
            self.test_data_per_solution[p] = test_data_per_classifier
            self.solution_dict[p] = solution
