import os
from typing import List, Dict
import mlflow
from mlflow.tracking import MlflowClient
from beir.retrieval.models import OnnxBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


class Experiment(object):
    def __init__(self, datasets: List[str],
                 datasets_path: str,
                 onnx_model: OnnxBERT,
                 batch_size: int,
                 score_function: str,
                 mlflow_configs: Dict[str, str]):
        self.dataset_paths = []
        for dataset in datasets:
            dataset_folder = os.path.join(datasets_path, dataset)
            self.dataset_paths.append(dataset_folder)
        self.onnx_model = onnx_model
        self.bs = batch_size
        self.score_func = score_function
        self.mlflow_configs = mlflow_configs
        self.__setup_experiment()
        self.__setup_mlflow()

    def __setup_experiment(self):
        self.model = DRES(self.onnx_model, batch_size=self.bs)
        self.retriever = EvaluateRetrieval(self.model, score_function=self.score_func)

    def __setup_mlflow(self):
        mlflow.set_tracking_uri(uri=self.mlflow_configs['tracking_uri'])
        self.client = MlflowClient()
        mlflow_experiment = self.client.get_experiment_by_name(name=self.mlflow_configs['experiment_name'])
        if mlflow_experiment is None:
            self.experiment_id = self.client.create_experiment(name=self.mlflow_configs['experiment_name'])
        else:
            if dict(mlflow_experiment)['lifecycle_stage'] == 'deleted':
                self.client.restore_experiment(dict(mlflow_experiment)['experiment_id'])
            self.experiment_id = dict(mlflow_experiment)['experiment_id']

    def experiemnt_pipeline(self):
        for dataset in self.dataset_paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=dataset).load(split='test')
            results = self.retriever.retrieve(corpus=corpus, queries=queries)
            ndcg, _map, recall, precision = self.retriever.evaluate(qrels=qrels,
                                                                    results=results,
                                                                    k_values=self.retriever.k_values)
            ndcg = self.__rename_metrics(metric_score=ndcg)
            _map = self.__rename_metrics(metric_score=_map)
            recall = self.__rename_metrics(metric_score=recall)
            precision = self.__rename_metrics(metric_score=precision)
            print('Results for', dataset)
            self.__tract_metric(dataset=dataset,
                                metric_name='ndcg',
                                metric_score=ndcg)
            print('NDCG:', ndcg)
            self.__tract_metric(dataset=dataset,
                                metric_name='recall',
                                metric_score=recall)
            print("Recall:", recall)
            self.__tract_metric(dataset=dataset,
                                metric_name='precision',
                                metric_score=precision)
            print('Precision:', precision)
            self.__tract_metric(dataset=dataset,
                                metric_name='map',
                                metric_score=_map)
            print('MAP:', _map)

    def __tract_metric(self,
                       dataset: str,
                       metric_name: str,
                       metric_score: Dict[str, float]):
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=dataset + '_' + metric_name):
            mlflow.log_metrics(metric_score)

    def __rename_metrics(self, metric_score: Dict[str, float]) -> Dict[str, float]:
        renamed_metric = {}
        for metric, score in metric_score.items():
            renamed_metric[metric.replace('@', '_')] = score
        return renamed_metric
