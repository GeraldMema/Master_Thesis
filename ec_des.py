import logging
from datetime import datetime

logging.basicConfig(filename='logs/ec_des.log', level=logging.INFO)
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

import yaml
from src.Evolutionary_Classification import Evolutionary_Classification
from src.EvaluationReport.Error_Analysis import Error_Analysis
from src.EvaluationReport.Visualization import Visualization
from src.MultipleClassificationModels.Classifiers import Classifiers
from src.DataProcessing.Data import Data
import os


def evolutionary_classification():
    if need_error_analysis:
        path = param_path
    else:
        path = run_path
    process = Evolutionary_Classification(cfg)
    report = process.apply_evolutionary_classification(c, data, path)
    fitness = report.process_results(cfg)
    return fitness


if __name__ == "__main__":
    # set up logging
    logging.info("*********** EVOLUTIONARY CLASSIFICATION WITH DYNAMIC ENSEMBLE SELECTION ********* ")

    # load the configurations
    with open('config.yml', 'r') as yml_file:
        cfg = yaml.safe_load(yml_file)

    # load all the configurations
    with open('config_all.yml', 'r') as yml_file:
        cfg_all = yaml.safe_load(yml_file)

    # Version
    logging.info("Version: " + str(cfg['project_params']['version']))

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Report Params
    timestamp = datetime.now()
    folder_name = "Computational_Results_" + timestamp.strftime("%d-%b-%y")
    folder_path = os.path.join(cfg['evaluation_params']['path'], folder_name)
    save_path = os.path.join(folder_path, "Results_" + timestamp.strftime("%d-%b-%y_%H-%M-%S"))

    # Runs
    runs = int(cfg['project_params']['execution_runs'])

    # create folder
    if not os.path.exists(ROOT_DIR + '\\' + save_path):
        os.makedirs(ROOT_DIR + '\\' + save_path)
    os.chdir(ROOT_DIR + '\\' + save_path)
    # Error analysis params
    need_error_analysis = cfg['evaluation_params']['error_analysis']
    # Error Analysis
    params_for_error_analysis = {}
    if need_error_analysis:
        ea = Error_Analysis(cfg['evaluation_params'])
        params_for_error_analysis = ea.params_for_error_analysis
        runs = 1

    # Get the classifiers
    c = Classifiers(cfg['multiple_classification_models_params'])
    # Get the data
    logging.info("Get and Process the Data")
    data = Data(cfg['data_params'])
    data.process(ROOT_DIR)

    # If error analysis is needed
    if need_error_analysis:

        for ea_param in params_for_error_analysis:  # for each param in error analysis
            error_analysis_path = os.path.join(save_path, 'Error Analysis for ' + ea_param)
            # create folder
            if not os.path.exists(ROOT_DIR + '\\' + error_analysis_path):
                os.makedirs(ROOT_DIR + '\\' + error_analysis_path)
            os.chdir(ROOT_DIR + '\\' + error_analysis_path)
            best_score_progress = []
            ea_param_values = cfg_all[params_for_error_analysis[ea_param]][ea_param]

            for val in ea_param_values:  # for each value in this param
                param_path = os.path.join(error_analysis_path, 'Error Analysis for ' + ea_param + 'with value ' + str(val))
                # create folder
                if not os.path.exists(ROOT_DIR + '\\' + param_path):
                    os.makedirs(ROOT_DIR + '\\' + param_path)
                os.chdir(ROOT_DIR + '\\' + param_path)
                cfg[params_for_error_analysis[ea_param]][ea_param] = val
                fitness_score = evolutionary_classification()
                best_score_progress.append(fitness_score)
                os.chdir(ROOT_DIR + '\\' + param_path)
            # Plot score per error analysis param
            plt_error_analysis = Visualization()
            plt_error_analysis.plot_error_analysis(best_score_progress, ea_param_values, ea_param)
            plt_error_analysis.error_analysis.savefig(
                os.path.join(ROOT_DIR + '\\' + save_path + '\\' + 'Best Score Per ' + ea_param + ".png"))
            os.chdir(ROOT_DIR + '\\' + error_analysis_path)

    else:
        for run in range(runs):
            logging.info('Execution Run: ' + str(run))
            run_path = os.path.join(save_path, 'Run ' + str(run))
            # create folder
            if not os.path.exists(ROOT_DIR + '\\' + run_path):
                os.makedirs(ROOT_DIR + '\\' + run_path)
            os.chdir(ROOT_DIR + '\\' + run_path)
            evolutionary_classification()
            os.chdir(ROOT_DIR + '\\' + run_path)

    os.chdir(ROOT_DIR + '\\' + save_path)
