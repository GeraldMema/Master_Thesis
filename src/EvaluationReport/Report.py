import yaml
import os


class Report:

    def __init__(self, evaluation_results, best_solution, total_exec_time, c):
        self.evaluation_results = evaluation_results
        self.best_solution = best_solution
        self.total_exec_time = total_exec_time
        self.classifiers = c

    def process_results(self, cfg):
        file_name = str(self.best_solution.fitness_score) + ".txt "

        file_handler = open(file=file_name, mode="w")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write("General Information:" + "\n")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write(f"The Best Solution Score is: {self.best_solution.fitness_score}" + "\n")
        file_handler.write(
            f"The Best Solution Score come from [{self.best_solution.population_producer}] population" + "\n")
        file_handler.write(f"Total time to find the best solution: {self.total_exec_time}" + "\n")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write(f"My Algorithm score: {self.evaluation_results['MY_ALG'][0]}" + "\n")
        file_handler.write(f"My Algorithm execution time: {self.evaluation_results['MY_ALG'][1]}" + "\n")
        for comp_clf in self.classifiers.comparison_classifiers:
            file_handler.write(f"{comp_clf} score: {self.evaluation_results[comp_clf][0]}" + "\n")
            file_handler.write(f"{comp_clf} execution time: {self.evaluation_results[comp_clf][1]}" + "\n")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write("Evolutionary Learning Params" + "\n")
        file_handler.write(f"Population Size: {cfg['evolutionary_learning_params']['population_size']}" + "\n")
        file_handler.write(
            f"Crossover Percentage: {cfg['evolutionary_learning_params']['crossover_percentage']}" + "\n")
        file_handler.write(
            f"Mutation Percentage: {cfg['evolutionary_learning_params']['mutation_percentage']}" + "\n")
        file_handler.write(
            f"Mutation Rate: {cfg['evolutionary_learning_params']['mutation_rate']}" + "\n")
        file_handler.write(
            f"Max Generations: {cfg['evolutionary_learning_params']['max_generations']}" + "\n")
        file_handler.write(
            f"Fitness Lambda (Accuracy Diversity trade off): {cfg['evolutionary_learning_params']['fitness_lambda']}" + "\n")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write("Evolutionary Learning Methods" + "\n")
        file_handler.write(
            f"Chromosome Representation: {cfg['evolutionary_learning_methods']['chromosome_representation']}" + "\n")
        file_handler.write(
            f"Crossover Method: {cfg['evolutionary_learning_methods']['crossover_methods']}" + "\n")
        file_handler.write(
            f"Mutation Method: {cfg['evolutionary_learning_methods']['mutation_methods']}" + "\n")
        file_handler.write(
            f"Parent Selection Method: {cfg['evolutionary_learning_methods']['parent_selection_method']}" + "\n")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write("Multiple Classification Models Params" + "\n")
        file_handler.write(
            f"Fusion Method: {cfg['multiple_classification_models_params']['fusion_method']}" + "\n")
        file_handler.write(
            f"Heterogeneous Classification: {cfg['multiple_classification_models_params']['heterogeneous_classification']}" + "\n")
        file_handler.write(
            f"Cross Validation: {cfg['multiple_classification_models_params']['cross_val']}." + "\n")
        file_handler.write(
            f"Fitness Score Metric: {cfg['multiple_classification_models_params']['fitness_score_metric']}" + "\n")
        file_handler.write(
            f"Selected Classifiers: {cfg['multiple_classification_models_params']['selected_classifiers']}" + "\n")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write("Dynamic Ensemble Selection Params" + "\n")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write("Data Params" + "\n")
        file_handler.write(
            "-----------------------------------------------------------------------------------------------------" + "\n")
        file_handler.write(
            f"Dataset: {cfg['data_params']['dataset']}" + "\n")
        file_handler.write(
            f"Dataset Normalization Method: {cfg['data_params']['normalization']}" + "\n")
        file_handler.write(
            "----------------------------------CLASSIFIERS GUESS VS PREDICTIONS-------------------------------------------------" + "\n")
        for useful_info in self.evaluation_results['MY_ALG'][2].useful_info:
            file_handler.write(f"Different Predictions: {self.evaluation_results['MY_ALG'][2].useful_info[useful_info][0]}" )
            file_handler.write(f"  Majority Voting: {self.evaluation_results['MY_ALG'][2].useful_info[useful_info][1]}")
            file_handler.write(f"  Actual: {self.evaluation_results['MY_ALG'][2].useful_info[useful_info][2]}" + "\n")
        file_handler.close()

        return self.best_solution.fitness_score, self.evaluation_results['MY_ALG'][0]

