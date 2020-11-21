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
import os


def evolutionary_classification():
    process = Evolutionary_Classification(cfg)
    report, results = process.apply_evolutionary_classification(cfg, run)
    best_fitness, final_prediction = report.process_results(cfg)
    return best_fitness, final_prediction, results


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

    # Runs
    runs = int(cfg['project_params']['execution_runs'])
    # Error analysis params
    need_error_analysis = cfg['evaluation_params']['error_analysis']
    # Error Analysis
    params_for_error_analysis = {}
    if need_error_analysis:
        ea = Error_Analysis(cfg['evaluation_params'])
        params_for_error_analysis = ea.params_for_error_analysis
        runs = 1

    # Report Params
    timestamp = datetime.now()
    folder_name = "Computational_Results_" + timestamp.strftime("%d-%b-%y")
    folder_path = os.path.join(cfg['evaluation_params']['path'], folder_name)
    if need_error_analysis:
        save_path = os.path.join(folder_path, "Error Analysis - Results_" + timestamp.strftime("%d-%b-%y_%H-%M-%S"))
    else:
        save_path = os.path.join(folder_path, "Results_" + timestamp.strftime("%d-%b-%y_%H-%M-%S"))
    # create folder
    if not os.path.exists(ROOT_DIR + '\\' + save_path):
        os.makedirs(ROOT_DIR + '\\' + save_path)
    os.chdir(ROOT_DIR + '\\' + save_path)

    # If error analysis is needed
    if need_error_analysis:
        for run in range(runs):
            for ea_param in params_for_error_analysis:  # for each param in error analysis
                error_analysis_path = os.path.join(save_path, 'Param ' + ea_param)
                # create folder
                if not os.path.exists(ROOT_DIR + '\\' + error_analysis_path):
                    os.makedirs(ROOT_DIR + '\\' + error_analysis_path)
                os.chdir(ROOT_DIR + '\\' + error_analysis_path)
                # keep the best fitness score and the best prediction score
                best_fitness_score_progress = []
                final_preds_score_progress = []
                # get the param values from config_all
                ea_param_values = cfg_all[params_for_error_analysis[ea_param]][ea_param]
                # keep in cache the default value of this param
                default_val = cfg[params_for_error_analysis[ea_param]][ea_param]
                for val in ea_param_values:  # for each value in this param
                    param_path = os.path.join(error_analysis_path, 'Value ' + str(val))
                    # create folder for error analysis in this param
                    if not os.path.exists(ROOT_DIR + '\\' + param_path):
                        os.makedirs(ROOT_DIR + '\\' + param_path)
                    os.chdir(ROOT_DIR + '\\' + param_path)
                    cfg[params_for_error_analysis[ea_param]][ea_param] = val
                    # start the algorithm
                    best_fitness, final_prediction, _ = evolutionary_classification()
                    best_fitness_score_progress.append(best_fitness)
                    final_preds_score_progress.append(final_prediction)
                    # re-construct the default value
                    cfg[params_for_error_analysis[ea_param]][ea_param] = default_val
                    os.chdir(ROOT_DIR + '\\' + param_path)
                # Plot score per error analysis param
                os.chdir(ROOT_DIR + '\\' + error_analysis_path)
                plt_error_analysis = Visualization()
                plt_error_analysis.plot_error_analysis(best_fitness_score_progress, final_preds_score_progress,
                                                       ea_param_values, ea_param)
    else:
        # create the comparison results with other algorithms
        my_alg = []
        rf = []
        xgb = []
        gb = []
        dt = []
        for run in range(runs):
            logging.info('Execution Run: ' + str(run))
            run_path = os.path.join(save_path, 'Run ' + str(run))
            # create folder in which we save the algorithm results with the other algorithms
            if not os.path.exists(ROOT_DIR + '\\' + run_path):
                os.makedirs(ROOT_DIR + '\\' + run_path)
            os.chdir(ROOT_DIR + '\\' + run_path)
            # start the algorithm
            _, _, evaluation_results = evolutionary_classification()
            os.chdir(ROOT_DIR + '\\' + save_path)
            my_alg.append(evaluation_results['MY_ALG'][0])
            rf.append(evaluation_results['RF'][0])
            xgb.append(evaluation_results['XGBoost'][0])
            gb.append(evaluation_results['GBoost'][0])
            dt.append(evaluation_results['DT'][0])
        # plot results
        plt_scores = Visualization()
        plt_scores.plot_scores(my_alg, rf, xgb, gb, dt)

    os.chdir(ROOT_DIR + '\\' + save_path)
