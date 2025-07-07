import argparse
import os

import mlflow
from mlflow.tracking.request_header.registry import \
    _request_header_provider_registry

from beir_extensions.configs import load_configurations
from beir_extensions.experiments import Experiment
from beir_extensions.mlflow import get_or_create_experiment, \
    mlflow_flattening, CustomHeaderProvider


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config_file',
        '-cf',
        help='The path of the configuration file'
    )
    args = arg_parser.parse_args()
    cfg = load_configurations(path=args.config_file)

    _request_header_provider_registry.register(CustomHeaderProvider)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_id = get_or_create_experiment(name=cfg.mlflow.experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=cfg.mlflow.run_name
    ):
        mlflow.log_artifact(local_path=args.config_file)
        experiment = Experiment(
            datasets=dict(cfg.datasets),
            datasets_path='datasets',
            results_dir=os.path.join('results', cfg.mlflow.run_name),
            model_type=cfg.model.type,
            model_name_or_path=cfg.model.name_or_path,
            tokenizer_name_or_path=cfg.model.tokenizer_name_or_path,
            batch_size=cfg.model.batch_size,
            matryoshka_dim=cfg.model.matryoshka_dim,
            score_function=cfg.model.score_function,
            sep=None if 'sep' not in cfg.model else cfg.model.sep,
            cls=None if 'cls' not in cfg.model else cfg.model.cls,
        )
        results, result_paths = experiment.experiment_pipeline()

        # Log the results to MLflow
        results = mlflow_flattening(results)
        mlflow.log_metrics(metrics=results)
        for path in result_paths:
            mlflow.log_artifact(local_path=path)
