import yaml
from beir.retrieval import models
from experiments.experiment import Experiment

if __name__ == '__main__':
    with open('configs/distillroberta_pca/semantic_search_config.yaml') as config_file:
        dataset_configs = yaml.safe_load(config_file)
    mlflow_configs = dataset_configs['mlflow']

    ##Load the onnx model and conduct the experiment
    onnx_model = models.OnnxBERT(onnx_filename=dataset_configs['onnx_filename'],
                                 model_path=dataset_configs['model_path'])
    experiment = Experiment(datasets=dataset_configs['datasets'],
                            datasets_path=dataset_configs['datasets_path'],
                            batch_size=dataset_configs['batch_size'],
                            onnx_model=onnx_model,
                            score_function=dataset_configs['score_function'],
                            mlflow_configs=mlflow_configs)
    experiment.experiemnt_pipeline()