import argparse
import yaml
from beir.retrieval import models
from experiments.experiment import HNSWEMonoT5xperiment

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file',
                            '-cf',
                            help='The path of the config file')
    args = arg_parser.parse_args()
    config_path = args.config_file
    with open(config_path) as config_file:
        dataset_configs = yaml.safe_load(config_file)
    mlflow_configs = dataset_configs['mlflow']
    es_configs = dataset_configs['es']

    ##Load the onnx model and conduct the experiment
    onnx_model = models.OnnxBERT(onnx_filename=dataset_configs['onnx_filename'],
                                 model_path=dataset_configs['model_path'])
    experiment = HNSWEMonoT5xperiment(datasets=dataset_configs['datasets'],
                                      datasets_path=dataset_configs['datasets_path'],
                                      hnsw_batch_size=dataset_configs['batch_size'],
                                      ce_model=dataset_configs["cross_encoder"]["model_name"],
                                      ce_batch_size=dataset_configs["cross_encoder"]["batch_size"],
                                      onnx_model=onnx_model,
                                      score_function=dataset_configs['score_function'],
                                      mlflow_configs=mlflow_configs,
                                      top_k=dataset_configs['k'],
                                      token_false=dataset_configs["monot5"]["false_token"],
                                      token_true=dataset_configs["monot5"]["true_token"])
    experiment.experiment_pipeline()
