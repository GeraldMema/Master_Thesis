import logging
from datetime import datetime
import operator
import os
from pmlb import fetch_data, classification_dataset_names

PROJECT_PATH = os.path.abspath(os.getcwd())

logging.basicConfig(filename=PROJECT_PATH + '/logs/ec_des.log', level=logging.INFO)
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.CRITICAL)
# console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

import yaml
from src.Evolutionary_Classification import Evolutionary_Classification
from src.EvaluationReport.Error_Analysis import Error_Analysis
from src.EvaluationReport.Visualization import Visualization
import os
import traceback


def evolutionary_classification(cfg, classification_dataset, run):
    process = Evolutionary_Classification(cfg)
    report, results = process.apply_evolutionary_classification(cfg, run, classification_dataset)
    if report:
        final_prediction = report.process_results(cfg)
    else:
        final_prediction = None
        results = None
    return final_prediction, results


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
        # runs = 1

    # Report Params
    timestamp = datetime.now()
    folder_name = "Computational_Results_" + timestamp.strftime("%d-%b-%y")
    folder_path = os.path.join(cfg['evaluation_params']['path'], folder_name)
    if need_error_analysis:
        save_path = os.path.join(folder_path,
                                 cfg['data_params']['dataset'] + " Error Analysis - Results_" + timestamp.strftime(
                                     "%d-%b-%y_%H-%M-%S"))
    else:
        save_path = os.path.join(folder_path,
                                 cfg['data_params']['dataset'] + " Results_" + timestamp.strftime("%d-%b-%y_%H-%M-%S"))
    # create folder
    if not os.path.exists(ROOT_DIR + '/' + save_path):
        os.makedirs(ROOT_DIR + '/' + save_path)
    os.chdir(ROOT_DIR + '/' + save_path)

    # If error analysis is needed
    if need_error_analysis:
        best_res_dict = {}
        res_per_run = {}
        classification_dataset_names = [
            'GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1',
            'parity5+5',
            'car_evaluation',
            'Hill_Valley_with_noise',
            'crx',
            'churn',
            'biomed',
            'allhypo',
            'ionosphere'
        ]
        for dataset in classification_dataset_names:
            if dataset=='GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1':
                d = 'GAMETES'
            else:
                d = dataset
            for ea_param in params_for_error_analysis:  # for each param in error analysis

                error_analysis_path = os.path.join(save_path, 'Param ' + ea_param)
                # create folder
                if not os.path.exists(ROOT_DIR + '/' + error_analysis_path):
                    os.makedirs(ROOT_DIR + '/' + error_analysis_path)
                os.chdir(ROOT_DIR + '/' + error_analysis_path)
                # keep the best fitness score and the best prediction score
                best_fitness_score_progress = []
                final_preds_score_progress = []
                eval_results = []
                # get the param values from config_all
                ea_param_values = cfg_all[params_for_error_analysis[ea_param]][ea_param]
                # keep in cache the default value of this param
                default_val = cfg[params_for_error_analysis[ea_param]][ea_param]
                for val in ea_param_values:  # for each value in this param
                    param_path = os.path.join(error_analysis_path, 'Value ' + str(val)[:5] + ' Dataset {}'.format(d[:5]))
                    # create folder for error analysis in this param
                    if not os.path.exists(ROOT_DIR + '/' + param_path):
                        os.makedirs(ROOT_DIR + '/' + param_path)
                    os.chdir(ROOT_DIR + '/' + param_path)
                    cfg[params_for_error_analysis[ea_param]][ea_param] = val
                    # start the algorithm
                    final_prediction, results = evolutionary_classification(cfg, dataset, 0)
                    if not final_prediction:
                        continue
                    if param_path not in best_res_dict:
                        best_res_dict[param_path] = final_prediction
                    else:
                        best_res_dict[param_path] += final_prediction
                    # best_fitness_score_progress.append(best_fitness)
                    final_preds_score_progress.append(final_prediction)
                    eval_results.append(results)
                    # re-construct the default value
                    cfg[params_for_error_analysis[ea_param]][ea_param] = default_val
                    os.chdir(ROOT_DIR + '/' + param_path)
                # Plot score per error analysis param
                os.chdir(ROOT_DIR + '/' + error_analysis_path)
                vis = Visualization()
                vis.plot_error_analysis(best_fitness_score_progress, final_preds_score_progress,
                                        ea_param_values, ea_param, d)
                vis.plot_error_analysis_all(eval_results, ea_param_values, ea_param, d)

                res_per_run[dataset] = eval_results
        # sort by fitness values
        sorted_fitness = dict(sorted(best_res_dict.items(), key=operator.itemgetter(1), reverse=True))
        # print('Best score (' + str(list(sorted_fitness.values())[0] / runs) + ') in ' + str(
        #     list(sorted_fitness.keys())[0]))
    else:

        # create the comparison results with other algorithms
        my_alg = []
        my_alg_not_GA = []
        rf = []
        xgb = []
        gb = []
        dt = []
        # for run in range(runs):
        run_all_datasets = cfg['data_params']['run_all_datasets']
        runs = 1
        best_datasets = ['car_evaluation']
        if run_all_datasets:
            for classification_dataset in classification_dataset_names:
                # print(classification_dataset, (classification_dataset not in best_datasets))
                if classification_dataset not in best_datasets:
                    # print(classification_dataset)
                    continue
                # else:
                #     df = fetch_data(classification_dataset)
                #     print("Analysis of {}".format(classification_dataset))
                #     print("Shape: {}".format(df.shape))
                #     print("Imbalance: {}".format(df["target"].value_counts()))
                #     print("Number of Classes: {}".format(len(df.target.unique())))
                #     print(df.describe())
                #     continue
                # create the comparison results with other algorithms
                my_alg = []
                my_alg_not_GA = []
                rf = []
                xgb = []
                gb = []
                dt = []
                for run in range(runs):
                    shape = fetch_data(classification_dataset, return_X_y=True)[0].shape
                    print('Start with {} dataset - {} run {}'.format(classification_dataset, shape,run))

                    run_path = os.path.join(save_path, 'Run {} '.format(run) + classification_dataset)
                    # create folder in which we save the algorithm results with the other algorithms
                    if not os.path.exists(ROOT_DIR + '/' + run_path):
                        os.makedirs(ROOT_DIR + '/' + run_path)
                    os.chdir(ROOT_DIR + '/' + run_path)
                    # start the algorithm
                    try:
                        _, evaluation_results = evolutionary_classification(cfg, classification_dataset, run)
                    except:
                        print('Problem with {}'.format(classification_dataset))
                        print(traceback.print_exc())
                        continue
                    if not evaluation_results:
                        continue
                    os.chdir(ROOT_DIR + '/' + save_path)
                    my_alg.append(evaluation_results['MY_ALG'][0])
                    my_alg_not_GA.append(evaluation_results['MY_ALG_WITHOUT_GA'][0])
                    rf.append(evaluation_results['RF'][0])
                    xgb.append(evaluation_results['XGBoost'][0])
                    gb.append(evaluation_results['GBoost'][0])
                    dt.append(evaluation_results['DT'][0])
                # plot results
                # vis = Visualization()
                # vis.plot_scores(my_alg, my_alg_not_GA, rf, xgb, gb, dt)
        else:
            for run in range(runs):
                # logging.info('Execution Run: ' + str(run))
                run_path = os.path.join(save_path, 'Run ' + str())
                # create folder in which we save the algorithm results with the other algorithms
                if not os.path.exists(ROOT_DIR + '/' + run_path):
                    os.makedirs(ROOT_DIR + '/' + run_path)
                os.chdir(ROOT_DIR + '/' + run_path)
                # start the algorithm
                _, evaluation_results = evolutionary_classification(cfg, str(run))
                if not evaluation_results:
                    continue
                os.chdir(ROOT_DIR + '/' + save_path)
                my_alg.append(evaluation_results['MY_ALG'][0])
                my_alg_not_GA.append(evaluation_results['MY_ALG_WITHOUT_GA'][0])
                rf.append(evaluation_results['RF'][0])
                xgb.append(evaluation_results['XGBoost'][0])
                gb.append(evaluation_results['GBoost'][0])
                dt.append(evaluation_results['DT'][0])
        # # plot results
        # vis = Visualization()
        # vis.plot_scores(my_alg, my_alg_not_GA, rf, xgb, gb, dt)

        # check if our result is better than random forest
        # for run in range(runs):
        #     if (my_alg[run] >= rf[run]) and (my_alg[run] >= xgb[run]) and (my_alg[run] >= gb[run]):
        #         print('Good Result in Run: ', run, ' Result: ', my_alg[run])
