import argparse
import yaml
from beir.retrieval import models
from experiments.experiment import Experiment

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

    ##Load the onnx model and conduct the experiment
    onnx_model = models.OnnxBGE(onnx_filename=dataset_configs['onnx_filename'],
                                model_path=dataset_configs['model_path'],
                                enable_query_instruction=dataset_configs["bge"]["enable_query_instruction"],
                                query_instruction=dataset_configs["bge"]["query_instruction"],
                                cls=dataset_configs["bge"]["cls"])
    experiment = Experiment(datasets=dataset_configs['datasets'],
                            datasets_path=dataset_configs['datasets_path'],
                            batch_size=dataset_configs['batch_size'],
                            onnx_model=onnx_model,
                            score_function=dataset_configs['score_function'],
                            mlflow_configs=mlflow_configs)
    experiment.experiment_pipeline()
