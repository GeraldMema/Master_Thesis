import logging
logging.basicConfig(filename='logs/ec_des.log', level=logging.INFO)
import yaml
from src.Evolutionary_Classification import Evolutionary_Classification

if __name__ == "__main__":
    # set up logging
    logging.info("*********** EVOLUTIONARY CLASSIFICATION WITH DYNAMIC ENSEMBLE SELECTION ********* ")

    # load the configurations
    with open('config.yml', 'r') as yml_file:
        cfg = yaml.safe_load(yml_file)

    # Version
    logging.info("Version: " + str(cfg['project_params']['version']))

    # Evolutionary Learning Process
    runs = int(cfg['project_params']['execution_runs'])
    for run in range(runs):
        logging.info('Execution Run: ' + str(run))
        process = Evolutionary_Classification(cfg)
        process.apply_evolutionary_clustering()
