def error_analysis_params():
    return {
        'lambda_factor': 'evolutionary_learning_params',
        'population_size': 'evolutionary_learning_params',
        'crossover_percentage': 'evolutionary_learning_params',
        'mutation_percentage': 'evolutionary_learning_params',
        'mutation_rate': 'evolutionary_learning_params',
        'max_generations': 'evolutionary_learning_params',
        'fitness_lambda': 'evolutionary_learning_params',
        'crossover_methods': 'evolutionary_learning_methods',
        'mutation_methods': 'evolutionary_learning_methods',
        'parent_selection_method': 'evolutionary_learning_methods',
        'dataset': 'data_params',
        'normalization': 'data_params'
    }


class Error_Analysis:

    def __init__(self, params):
        self.selected_params_for_error_analysis = params['error_analysis_params']
        self.possible_params_for_error_analysis = error_analysis_params()
        self.params_for_error_analysis = self.get_params()

    def get_params(self):
        params_for_error_analysis = {}
        for pos_param in self.possible_params_for_error_analysis:
            if pos_param in self.selected_params_for_error_analysis:
                params_for_error_analysis[pos_param] = self.possible_params_for_error_analysis[pos_param]
        return params_for_error_analysis


    def process(self,cfg):
        """
        Add Description
        """
        param_list = cfg[self.parameter_path_analysis][self.parameter_name_analysis]



