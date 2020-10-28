class Classifiers_Data:
    """
    Add Description
    """

    def __init__(self, data, population):
        self.data = data
        self.pop = population

    def extract_data_per_solution(self):
        """
        This function will return a list of dicts with all the needed data info for each classifier. The
        list represent all the current solutions that we have in a population. The dict will contain the classifier name
        as key and a pandas dataframe with the subset of selected features.

        :param
        :return:
        """

        population_size = self.pop.current_pop.shape[0]
        solution_representation = self.pop.solution_representation
        classifiers_dict = self.pop.solution_representation.solution_info.classifiers_dict
        features_dict = self.pop.solution_representation.solution_info.features_dict

        data_per_solution = []
        for p in range(population_size):
            solution = self.pop.current_pop[p]
            data_per_classifier = {}
            if self.pop.solution_representation.representation_method == '1D':
                selected_classifiers = None
                selected_features = None
            if self.pop.solution_representation.representation_method == '2D':
                selected_classifiers = None
                selected_features = None
            if self.pop.solution_representation.representation_method == 'dual':
                selected_classifiers = None
                selected_features = None
            for clf_indx in selected_classifiers:
                classifier_name = classifiers_dict[clf_indx]
                data = self.data.train_data[selected_features]
                data_per_classifier[classifier_name] = data
            data_per_solution.append(data_per_classifier)

        return data_per_solution

    def get_models_per_classifiers(self):
        """
        initialize models for each classifier
        """
