from src.data_utility import delete_directory, prepare_data, distribute_classes, train_submodels, ism_post_process, evaluate_ism
from src import utility_functions
import json
import os
from os.path import join
from datetime import datetime
import sqlite3

def main():
    """
    Main function for running the ISM (Independent Softmax Model) pipeline.

    This function orchestrates the entire ISM pipeline, including data preparation,
    submodel training, post-processing, and evaluation.

    Returns:
        None
    """    
    # Load the configuration file
    with open("./config/config.json", "r") as config_file:
        config = json.load(config_file)

    dataset_name = config['dataset_name']
    overwrite = config['overwrite']
    m = config['distance_measure']
    method = "ISM"


    # Loop through different combinations of n_classes and n_clusters
    # for n_classes, n_clusters in zip([5000, 5000, 7500, 7500, 7500, 8900, 8900, 8900], [3,4, 2, 3, 4, 2, 3, 4]): #zip([1000, 5000, 6500, 7500, 8000, 8900] , [2, 2, 3, 3, 3, 4]): #, 10000, 20000, 50000], [3,5,10,20]): #zip([1000, 5000, 10000, 20000, 50000], [2,3,5,10,20]):
    for n_classes, n_clusters in zip([5000], [3]):
        for iter in range(10):
            conn = sqlite3.connect("./results/results.db")
            cursor = conn.cursor()

            scenario = str(n_classes) + '_' + method + str(n_clusters)
            data_scenario_path = join(config[dataset_name]["scenario_embs"], scenario)
            model_scenario_path = join(config[dataset_name]["scenario_submodels"], scenario)
            super_scenario_path = join(config[dataset_name]["scenario_embs"], str(n_classes))

            # Delete existing data and model directories
            delete_directory(data_scenario_path)
            delete_directory(model_scenario_path)

            # Insert a new row into the results database
            cursor.execute("INSERT INTO results(iteration, n_classes, n_clusters, dataset_name, ism_start_timestamp) \
                            VALUES (?, ?, ?, ?, ?)", (iter, n_classes, n_clusters, dataset_name, datetime.now()))

            conn.commit()
            conn.close()

            utility_functions.pprint(("-------------------------------"), dataset_name)
            utility_functions.pprint(("dataset_name = ", dataset_name), dataset_name)
            utility_functions.pprint(("meth = ", method), dataset_name)
            utility_functions.pprint(("n_classes = ", n_classes), dataset_name)
            utility_functions.pprint(("n_clusters = ", n_clusters), dataset_name)

            # Create necessary directories
            os.makedirs(super_scenario_path, exist_ok=True)
            os.makedirs(data_scenario_path, exist_ok=True)
            os.makedirs(model_scenario_path, exist_ok=True)

            # Prepare data, distribute classes, and train submodels
            trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall = prepare_data(n_classes)
            parts = distribute_classes(method, n_classes, n_clusters, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall)
            train_submodels(method, n_classes, parts, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall)

            # Perform post-processing and evaluation on the test set
            test_softmax_classes = ism_post_process(m, n_classes, parts, testx, 'test')
            evaluate_ism(iter, m, n_classes, n_clusters, testl, test_softmax_classes)

if __name__ == "__main__":
    main()