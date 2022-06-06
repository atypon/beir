import os
from typing import List, Dict
import mlflow
from mlflow.tracking import MlflowClient
from beir.retrieval.models import OnnxBERT
from beir.retrieval.search.lexical import BM25Search as BM25
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
            try:
                corpus, queries, qrels = GenericDataLoader(data_folder=dataset).load(split='test')
                results = self.retriever.retrieve(corpus=corpus, queries=queries)
                ndcg, _map, recall, precision = self.retriever.evaluate(qrels=qrels,
                                                                        results=results,
                                                                        k_values=self.retriever.k_values)
                ndcg = self.__rename_metrics(metric_score=ndcg)
                _map = self.__rename_metrics(metric_score=_map)
                recall = self.__rename_metrics(metric_score=recall)
                precision = self.__rename_metrics(metric_score=precision)
                flatten_metrics = self.__concat_metrics(ndcg=ndcg,
                                                        recall=recall,
                                                        _map=_map,
                                                        precision=precision)
                self.__tract_metric(dataset=dataset, metric_score=flatten_metrics)
                print('Results for', dataset)
                print('NDCG:', ndcg)
                print("Recall:", recall)
                print('Precision:', precision)
                print('MAP:', _map)
            except:
                print('There is an error in this dataset:', dataset)

    def __track_metric(self,
                       dataset: str,
                       metric_score: Dict[str, float]):
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=dataset):
            mlflow.log_metrics(metric_score)

    def __rename_metrics(self, metric_score: Dict[str, float]) -> Dict[str, float]:
        renamed_metric = {}
        for metric, score in metric_score.items():
            renamed_metric[metric.replace('@', '_')] = score
        return renamed_metric

    def __concat_metrics(self,
                         ndcg: Dict[str, float],
                         recall: Dict[str, float],
                         _map: Dict[str, float],
                         precision: Dict[str, float]) -> Dict[str, float]:
        flatten_metrics = {}
        for metric in (ndcg, recall, _map, precision):
            flatten_metrics.update(metric)
        return flatten_metrics

class RerankExperiment(Experiment):
    def __init__(self,
                 datasets: List[str],
                 datasets_path: str,
                 onnx_model: OnnxBERT,
                 batch_size: int,
                 top_k: int,
                 score_function: str,
                 es_hostname: str,
                 initialize: bool,
                 mlflow_configs: Dict[str, str]):
        self.k = top_k
        self.es_hostname = es_hostname
        self.initialize = initialize
        super().__init__(datasets, datasets_path, onnx_model, batch_size, score_function, mlflow_configs)

    def __create_bm25_retriever(self, index_name):
        model = BM25(index_name=index_name, hostname=self.es_hostname, initialize=self.initialize)
        retriever = EvaluateRetrieval(model)
        return retriever

    def experiemnt_pipeline(self):
        for dataset in self.dataset_paths:
            try:
                corpus, queries, qrels = GenericDataLoader(data_folder=dataset).load(split='test')
                index_name = dataset.replace('/', '_')
                bm25_retriever = self.__create_bm25_retriever(index_name=index_name)
                bm25_results = bm25_retriever.retrieve(corpus=corpus, queries=queries)
                rerank_results = self.retriever.rerank(corpus=corpus,
                                                       queries=queries,
                                                       results=bm25_results,
                                                       top_k=self.k)
                ndcg, _map, recall, precision = self.retriever.evaluate(qrels=qrels,
                                                                        results=rerank_results,
                                                                        k_values=self.retriever.k_values)
                ndcg = self.__rename_metrics(metric_score=ndcg)
                _map = self.__rename_metrics(metric_score=_map)
                recall = self.__rename_metrics(metric_score=recall)
                precision = self.__rename_metrics(metric_score=precision)
                flatten_metrics = self.__concat_metrics(ndcg=ndcg,
                                                        recall=recall,
                                                        _map=_map,
                                                        precision=precision)
                self.__track_metric(dataset=dataset, metric_score=flatten_metrics)
                print('Results for', dataset)
                print('NDCG:', ndcg)
                print("Recall:", recall)
                print('Precision:', precision)
                print('MAP:', _map)
            except:
                print('There is an error in this dataset:', dataset)
