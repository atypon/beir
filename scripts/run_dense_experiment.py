import argparse
import yaml

import mlflow

from beir.retrieval import models
from beir.extensions.experiments  import Experiment
from beir.extensions.mlflow import get_or_create_experiment, mlflow_flattening


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file',
                            '-cf',
                            help='The path of the config file')
    args = arg_parser.parse_args()
    with open(args.config_file) as config_file:
        cfg = yaml.safe_load(config_file)

    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    experiment_id = get_or_create_experiment(name=cfg['mlflow']['experiment_name'])

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=cfg['mlflow']['run_name']
    ):
        mlflow.log_artifact(artifact_path=args.config_file)
        # Load the onnx model and conduct the experiment
        onnx_model = models.OnnxBERT(onnx_filename=cfg['onnx_filename'],
                                    model_path=cfg['model_path'],
                                    matryoshka_dim=cfg['matryoshka_dim']
                                    )
        experiment = Experiment(datasets=cfg['datasets'],
                                datasets_path='datasets',
                                batch_size=cfg['batch_size'],
                                onnx_model=onnx_model,
                                score_function=cfg['score_function'],
                                run_name=cfg['mlflow']['run_name'])
        results, result_paths = experiment.experiment_pipeline()
        results = mlflow_flattening(results)
        mlflow.log_metrics(metrics=results)
        for path in result_paths:
            mlflow.log_artifact(local_path=path)
