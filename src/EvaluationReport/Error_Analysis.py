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
        'normalization': 'data_params',
        'diversity_only_error': 'evolutionary_learning_methods',
        'initialization_method': 'evolutionary_learning_methods',
        'fusion_method': 'multiple_classification_models_params',
        'my_rules': 'evolutionary_learning_methods',
        'max_depth': 'multiple_classification_models_params',
        'min_sample': 'multiple_classification_models_params',
        'learning_factor': 'evolutionary_learning_params',
        'lambda_factor_strategy': 'evolutionary_learning_methods',
        'one_minus_lambda': 'evolutionary_learning_methods',
        'set_lambda_after': 'evolutionary_learning_methods'
    }


class Error_Analysis:

    def __init__(self, params):
        self.selected_params_for_error_analysis = params['error_analysis_params']
        self.possible_params_for_error_analysis = error_analysis_params()
        self.params_for_error_analysis = self.get_params()

    def get_params(self):
        params_for_error_analysis = {}
        print(self.possible_params_for_error_analysis)
        for pos_param in self.possible_params_for_error_analysis:
            if pos_param in self.selected_params_for_error_analysis:
                params_for_error_analysis[pos_param] = self.possible_params_for_error_analysis[pos_param]
        return params_for_error_analysis




