from src.MultipleClassificationModels.Classifiers import Classifiers

class Classifiers_Data:
    """
    Add Description
    """

    def __init__(self, data, classifiers):
        self.classifiers = classifiers
        self.data = data
        self.pop = None
        self.train_data_per_solution = {}
        self.train_data_per_solution = {}
        self.cv_data_per_solution = {}
        self.test_data_per_solution = {}
        self.solution_dict = {}

    def extract_data_per_solution(self, status, solution_idx, population):
        """
        This function will return a list of dicts with all the needed data info for each classifier. The
        list represent all the current solutions that we have in a population. The dict will contain the classifier name
        as key and a pandas dataframe with the subset of selected features.

        :param
        :return:
        """

        self.pop = population
        if status == 'crossover':
            population_size = self.pop.crossover_pop.shape[0]
        elif status == 'mutation':
            population_size = self.pop.mutation_pop.shape[0]
        else:
            population_size = self.pop.current_pop.shape[0]
        classifiers_dict = self.classifiers.classifier_dict
        features_dict = self.data.features_dict

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
                features = [features_dict[idx] for idx, i in enumerate(solution) if i == 1]
                train_data_per_classifier[p] = self.data.X_train[features]
                cv_data_per_classifier[p] = self.data.X_cv[features]
                test_data_per_classifier[p] = self.data.X_test[features]
                self.train_data_per_solution[p + solution_idx] = train_data_per_classifier
                self.cv_data_per_solution[p + solution_idx] = cv_data_per_classifier
                self.test_data_per_solution[p + solution_idx] = test_data_per_classifier
                self.solution_dict[p + solution_idx] = solution
            if self.pop.solution_representation.representation_method == '2D':
                for idx, selected_features in enumerate(solution):
                    clf = classifiers_dict[idx]
                    features = [features_dict[idx] for idx, i in enumerate(selected_features) if i == 1]
                    train_data_per_classifier[clf] = self.data.X_train[features]
                    cv_data_per_classifier[clf] = self.data.X_cv[features]
                    test_data_per_classifier[clf] = self.data.X_test[features]
                self.train_data_per_solution[p + solution_idx] = train_data_per_classifier
                self.cv_data_per_solution[p + solution_idx] = cv_data_per_classifier
                self.test_data_per_solution[p + solution_idx] = test_data_per_classifier
                self.solution_dict[p + solution_idx] = solution
            if self.pop.solution_representation.representation_method == 'dual':
                # TODO: apply this based on the paper
                continue


    def extract_data_for_ensemble_1D(self, population, clfs):
        train_data_per_classifier = {}
        test_data_per_classifier = {}
        features_dict = self.data.features_dict
        for i in range(len(population)):
            features = [features_dict[idx] for idx, i in enumerate(population[i]) if i == 1]
            train_data_per_classifier[i] = self.data.X_train[features]
            test_data_per_classifier[i] = self.data.X_test[features]

        return train_data_per_classifier, test_data_per_classifier

    def extract_data_for_ensemble_2D(self, solution):
        classifiers_dict = self.classifiers.classifier_dict
        features_dict = self.data.features_dict
        train_data_per_classifier = {}
        test_data_per_classifier = {}
        for idx, selected_features in enumerate(solution):
            clf = classifiers_dict[idx]
            features = [features_dict[idx] for idx, i in enumerate(selected_features) if i == 1]
            test_data_per_classifier[clf] = self.data.X_test[features]
            train_data_per_classifier[clf] = self.data.X_train[features]

        return train_data_per_classifier, test_data_per_classifier


