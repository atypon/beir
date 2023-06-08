import os
from typing import List, Dict
import mlflow
from mlflow.tracking import MlflowClient
from beir.retrieval.models import OnnxBERT
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.dense import HNSWFaissSearch
from beir.reranking.models.cross_encoder import CrossEncoder
from beir.reranking.models.mono_t5 import MonoT5
from beir.reranking import Rerank
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

    def experiment_pipeline(self):
        for dataset in self.dataset_paths:
            try:
                corpus, queries, qrels = GenericDataLoader(data_folder=dataset).load(split='test')
                results = self.retriever.retrieve(corpus=corpus, queries=queries)
                self._eval_pipeline(qrels=qrels, results=results, dataset=dataset)
            except:
                print('There is an error in this dataset:', dataset)

    def _track_metric(self,
                      dataset: str,
                      metric_score: Dict[str, float]):
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=dataset):
            mlflow.log_metrics(metric_score)

    def _rename_metrics(self, metric_score: Dict[str, float]) -> Dict[str, float]:
        renamed_metric = {}
        for metric, score in metric_score.items():
            renamed_metric[metric.replace('@', '_')] = score
        return renamed_metric

    def _concat_metrics(self,
                        ndcg: Dict[str, float],
                        recall: Dict[str, float],
                        _map: Dict[str, float],
                        precision: Dict[str, float]) -> Dict[str, float]:
        flatten_metrics = {}
        for metric in (ndcg, recall, _map, precision):
            flatten_metrics.update(metric)
        return flatten_metrics

    def _eval_pipeline(self,
                       qrels: Dict[str, Dict[str, int]],
                       results: Dict[str, Dict[str, float]],
                       dataset: str) -> None:
        """
        Evaluation of the results of a pipeline and log them in MLFlow
        :param qrels: the relevance of each query-doc pair
        :param results: the results of the pipeline
        :param dataset: the dataset name
        """
        ndcg, _map, recall, precision = self.retriever.evaluate(qrels=qrels,
                                                                results=results,
                                                                k_values=self.retriever.k_values)
        ndcg = self._rename_metrics(metric_score=ndcg)
        _map = self._rename_metrics(metric_score=_map)
        recall = self._rename_metrics(metric_score=recall)
        precision = self._rename_metrics(metric_score=precision)
        flatten_metrics = self._concat_metrics(ndcg=ndcg,
                                               recall=recall,
                                               _map=_map,
                                               precision=precision)
        self._track_metric(dataset=dataset, metric_score=flatten_metrics)
        print('Results for', dataset)
        print('NDCG:', ndcg)
        print("Recall:", recall)
        print('Precision:', precision)
        print('MAP:', _map)


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

    def _create_bm25_retriever(self, index_name) -> EvaluateRetrieval:
        model = BM25(index_name=index_name, hostname=self.es_hostname, initialize=self.initialize)
        retriever = EvaluateRetrieval(model)
        return retriever

    def experiment_pipeline(self):
        for dataset in self.dataset_paths:
            try:
                corpus, queries, qrels = GenericDataLoader(data_folder=dataset).load(split='test')
                index_name = dataset.replace('/', '_')
                bm25_retriever = self._create_bm25_retriever(index_name=index_name)
                bm25_results = bm25_retriever.retrieve(corpus=corpus, queries=queries)
                rerank_results = self.retriever.rerank(corpus=corpus,
                                                       queries=queries,
                                                       results=bm25_results,
                                                       top_k=self.k)
                self._eval_pipeline(qrels=qrels, results=rerank_results, dataset=dataset)
            except:
                print('There is an error in this dataset:', dataset)


class RerankBiCrossEncodersExperiment(RerankExperiment):
    """
    An extention class where the results of the RerankExperiment pipeline will be reranked based on a cross encoder
    """
    def __init__(self,
                 datasets: List[str],
                 datasets_path: str,
                 onnx_model: OnnxBERT,
                 ce_model: str,
                 bi_batch_size: int,
                 ce_batch_size: int,
                 top_k: int,
                 score_function: str,
                 es_hostname: str,
                 initialize: bool,
                 mlflow_configs: Dict[str, str]):
        """
        Initialize the class by load ing the models
        :param datasets: a list with the datasets to evaluate
        :param datasets_path: the path we stored the datasets
        :param onnx_model: the onnx bi-encoder
        :param ce_model: the hf card of the cross encoder model
        :param bi_batch_size: the batch size for the bi-encoder step.
        :param ce_batch_size: the batch size for the cross-encoder step
        :param top_k: retrieve top_k results using the bi-encoder
        :param score_function: the similarity metric
        :param es_hostname: the hostname of ElasticSearch
        :param initialize: a boolean to decide if we will initialize the ES or not
        :param mlflow_configs: the MLFlow server configurations
        """
        self.ce = CrossEncoder(ce_model)
        self.ce_batch_size = ce_batch_size
        self.reranker = Rerank(model=self.ce, batch_size=self.ce_batch_size)
        super().__init__(datasets=datasets,
                         datasets_path=datasets_path,
                         onnx_model=onnx_model,
                         top_k=top_k,
                         batch_size=bi_batch_size,
                         score_function=score_function,
                         es_hostname=es_hostname,
                         initialize=initialize,
                         mlflow_configs=mlflow_configs)

    def _rerank_pipeline(self,
                          corpus: Dict[str, Dict[str, str]],
                          queries: Dict[str, str],
                          index_name: str) \
            -> Dict[str, Dict[str, float]]:
        """
        perform all the rerank steps of the pipeline
        :param corpus: the corpus of a specific dataset
        :param queries: the queries of this dataset
        :param index_name: the name of the ES index
        :return  the reranked results.
        """
        bm25_retriever = self._create_bm25_retriever(index_name=index_name)
        bm25_results = bm25_retriever.retrieve(corpus=corpus, queries=queries)
        bi_rerank_results = self.retriever.rerank(corpus=corpus,
                                                  queries=queries,
                                                  results=bm25_results,
                                                  top_k=(2 * self.k))
        ce_rerank_results = self.reranker.rerank(corpus=corpus,
                                                 queries=queries,
                                                 results=bi_rerank_results,
                                                 top_k=self.k)
        return ce_rerank_results

    def experiment_pipeline(self):
        """
        The full pipeline of the experiment. The steps of this pipeline are:
        1) Retrieve documents using BM25
        2) Rerank them using the embeddings extracted by a bi-encoder
        3) Rerank the previous results using a cross encoder
        Finally, evaluate the results and log the metrics to MLFlow server.
        """
        for dataset in self.dataset_paths:
            try:
                corpus, queries, qrels = GenericDataLoader(data_folder=dataset).load(split='test')
                index_name = dataset.replace('/', '_')
                rerank_results = self._rerank_pipeline(corpus=corpus, queries=queries, index_name=index_name)
                self._eval_pipeline(qrels=qrels, results=rerank_results, dataset=dataset)
            except:
                print('There is an error in this dataset:', dataset)


class BM25CrossEncoderExperiment(RerankBiCrossEncodersExperiment):
    """
    BM25 + CE rerank experiment
    """
    def __init__(self,
                 datasets: List[str],
                 datasets_path: str,
                 onnx_model: OnnxBERT,
                 ce_model: str,
                 bi_batch_size: int,
                 ce_batch_size: int,
                 top_k: int,
                 score_function: str,
                 es_hostname: str,
                 initialize: bool,
                 mlflow_configs: Dict[str, str]):
        """
        Initialize the class by load ing the models
        :param datasets: a list with the datasets to evaluate
        :param datasets_path: the path we stored the datasets
        :param onnx_model: the onnx bi-encoder
        :param ce_model: the hf card of the cross encoder model
        :param bi_batch_size: the batch size for the bi-encoder step.
        :param ce_batch_size: the batch size for the cross-encoder step
        :param top_k: retrieve top_k results using the bi-encoder
        :param score_function: the similarity metric
        :param es_hostname: the hostname of ElasticSearch
        :param initialize: a boolean to decide if we will initialize the ES or not
        :param mlflow_configs: the MLFlow server configurations
        """
        super().__init__(datasets=datasets,
                         datasets_path=datasets_path,
                         onnx_model=onnx_model,
                         ce_model=ce_model,
                         bi_batch_size=0,
                         ce_batch_size=ce_batch_size,
                         top_k=top_k,
                         score_function=score_function,
                         es_hostname=es_hostname,
                         initialize=initialize,
                         mlflow_configs=mlflow_configs)

    def _rerank_pipeline(self,
                         corpus: Dict[str, Dict[str, str]],
                         queries: Dict[str, str],
                         index_name: str) -> Dict[str, Dict[str, float]]:
        """
               perform all the rerank steps of the pipeline
               :param corpus: the corpus of a specific dataset
               :param queries: the queries of this dataset
               :param index_name: the name of the ES index
               :return  the reranked results.
               """
        bm25_retriever = self._create_bm25_retriever(index_name=index_name)
        bm25_results = bm25_retriever.retrieve(corpus=corpus, queries=queries)
        ce_rerank_results = self.reranker.rerank(corpus=corpus,
                                                 queries=queries,
                                                 results=bm25_results,
                                                 top_k=self.k)
        return ce_rerank_results


class HNSWExperiment(Experiment):
    """
    A class that represents the experiment where the first step is hswn and the second one a cross encoder
    """

    def __init__(self,
                 datasets: List[str],
                 datasets_path: str,
                 onnx_model: OnnxBERT,
                 ce_model: str,
                 hnsw_batch_size: int,
                 ce_batch_size: int,
                 top_k: int,
                 score_function: str,
                 mlflow_configs: Dict[str, str],
                 hnsw_store_n: int = 512,
                 hnsw_ef_search: int = 128,
                 hnsw_ef_construction: int = 200,
                 ):
        """
         Initialize the class by load ing the models
        :param datasets: a list with the datasets to evaluate
        :param datasets_path: the path we stored the datasets
        :param onnx_model: the onnx bi-encoder
        :param ce_model: the hf card of the cross encoder model
        :param hnsw_batch_size: the batch size for the bi-encoder step in hswn algorithm.
        :param ce_batch_size: the batch size for the cross-encoder step
        :param top_k: retrieve top_k results using the bi-encoder
        :param score_function: the similarity metric
        """
        self.k = top_k
        super().__init__(datasets=datasets,
                         datasets_path=datasets_path,
                         onnx_model=onnx_model,
                         batch_size=hnsw_batch_size,
                         score_function=score_function,
                         mlflow_configs=mlflow_configs)
        self.hnsw_store_n = hnsw_store_n
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        self.ce = CrossEncoder(model_path=ce_model)
        self.ce_batch_size = ce_batch_size
        self.reranker = Rerank(model=self.ce, batch_size=self.ce_batch_size)

    def _rerank_pipeline(self,
                         corpus: Dict[str, Dict[str, str]],
                         queries: Dict[str, str]) \
            -> Dict[str, Dict[str, float]]:
        """
        perform all the rerank steps of the pipeline.
        :param corpus: the corpus of a specific dataset
        :param queries: the queries of this dataset
        :return  the reranked results.
        """
        faiss_search = HNSWFaissSearch(self.onnx_model,
                                       batch_size=self.bs,
                                       hnsw_store_n=self.hnsw_store_n,
                                       hnsw_ef_search=self.hnsw_ef_search,
                                       hnsw_ef_construction=self.hnsw_ef_construction)
        self.retriever = EvaluateRetrieval(faiss_search, score_function=self.score_func)
        faiss_results = self.retriever.retrieve(corpus=corpus, queries=queries)
        ce_rerank_results = self.reranker.rerank(corpus=corpus,
                                                 queries=queries,
                                                 results=faiss_results,
                                                 top_k=self.k)
        return ce_rerank_results

    def experiment_pipeline(self):
        """
        The full pipeline of the experiment. The steps of this pipeline are:
        1) Index the documents by their embeddings extracted by a bi-encoder in faiss
        2) Rerank the previous results using a cross encoder
        Finally, evaluate the results and log the metrics to MLFlow server.
        """
        for dataset in self.dataset_paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=dataset).load(split='test')
            rerank_results = self._rerank_pipeline(corpus=corpus, queries=queries)
            self._eval_pipeline(qrels=qrels, results=rerank_results, dataset=dataset)


class HNSWEMonoT5xperiment(Experiment):
    """
    A class that represents the experiment where the first step is hswn and the second one a cross encoder
    """

    def __init__(self,
                 datasets: List[str],
                 datasets_path: str,
                 onnx_model: OnnxBERT,
                 token_false: str,
                 token_true: str,
                 ce_model: str,
                 hnsw_batch_size: int,
                 ce_batch_size: int,
                 top_k: int,
                 score_function: str,
                 mlflow_configs: Dict[str, str],
                 hnsw_store_n: int = 512,
                 hnsw_ef_search: int = 128,
                 hnsw_ef_construction: int = 200,
                 ):
        """
         Initialize the class by load ing the models
        :param datasets: a list with the datasets to evaluate
        :param datasets_path: the path we stored the datasets
        :param onnx_model: the onnx bi-encoder
        :param token_false: the token that represents the false token
        :param token_true: the token that represents the true token
        :param ce_model: the hf card of the cross encoder model
        :param hnsw_batch_size: the batch size for the bi-encoder step in hswn algorithm.
        :param ce_batch_size: the batch size for the cross-encoder step
        :param top_k: retrieve top_k results using the bi-encoder
        :param score_function: the similarity metric
        """
        self.k = top_k
        super().__init__(datasets=datasets,
                         datasets_path=datasets_path,
                         onnx_model=onnx_model,
                         batch_size=hnsw_batch_size,
                         score_function=score_function,
                         mlflow_configs=mlflow_configs)
        self.hnsw_store_n = hnsw_store_n
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        self.ce = MonoT5(model_path=ce_model, token_true=token_true, token_false=token_false)
        self.ce_batch_size = ce_batch_size
        self.reranker = Rerank(model=self.ce, batch_size=self.ce_batch_size)
