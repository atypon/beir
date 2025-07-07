import json
import os
from typing import Literal

from beir.datasets.data_loader import GenericDataLoader
from beir.extensions.models.base_model import CustomModel
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models.cross_encoder import CrossEncoder
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from beir.extensions.models.onnx import OnnxModel
from beir.extensions.models.sentence_transformers import \
    SentenceTransformersModel


class Experiment(object):

    """
    A class that conducats the experiment on the desired datasets.
    """

    def __init__(
        self,
        datasets: dict[str, dict[str, str]],
        datasets_path: str,
        results_dir: str,
        model_type: Literal['onnx', 'sentence_transformers'],
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        batch_size: int,
        sep: str | None,
        cls: bool | None = None,
        matryoshka_dim: int | None = None,
        score_function: str = 'cos_sim',

    ):
        """
        Initialize the experiment with the datasets and model configurations.
        """
        self.datasets = datasets
        self.dataset_paths = []
        for dataset in datasets:
            dataset_folder = os.path.join(datasets_path, dataset)
            self.dataset_paths.append(dataset_folder)
        self.results_dir = results_dir
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.batch_size = batch_size
        self.sep = sep
        self.cls = cls
        self.matryoshka_dim = matryoshka_dim
        self.score_func = score_function

    def __setup_models_for_dataset(
        self,
        query_prompt: str | None = None,
        document_prompt: str | None = None
    ):
        """
        Sets up the model and retriever for the dataset experiment.
        """
        if self.model_type == 'onnx':
            self.model = OnnxModel(
                onnx_path=self.model_name_or_path,
                tokenizer_path=self.tokenizer_name_or_path,
                matryoshka_dim=self.matryoshka_dim,
                query_prompt=query_prompt,
                document_prompt=document_prompt,
                sep=self.sep,
                cls=self.cls
            )
        elif self.model_type == 'sentence_transformers':
            self.model = SentenceTransformersModel(
                model_path=self.model_name_or_path,
                matryoshka_dim=self.matryoshka_dim,
                query_prompt=query_prompt,
                document_prompt=document_prompt,
            )
        self.model = DRES(self.model, batch_size=self.batch_size)
        self.retriever = EvaluateRetrieval(
            self.model,
            score_function=self.score_func
        )

    def __setup_result_dir(self):
        """
        Creates directory named after the current run to save json
        file of results
        """
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)

    def experiment_pipeline(self) -> tuple[dict[str, dict[str, float]], str]:
        """
        Run the complete pipeline
        :return: dictionary of dictionaries containing metrics for
            each experiment along with paths with result files
        """
        metrics_per_dataset = {}
        results_paths = []
        for dataset, dataset_path in zip(self.datasets, self.dataset_paths):

            corpus, queries, qrels = GenericDataLoader(
                data_folder=dataset_path
            ).load(split='test')
            self.__setup_models_for_dataset(
                query_prompt=self.datasets[dataset]['query_instruction'],
                document_prompt=self.datasets[dataset]['document_instruction']
            )
            results = self.retriever.retrieve(corpus=corpus, queries=queries)
            metrics, results_path = self._eval_pipeline(
                qrels=qrels,
                results=results, dataset=dataset
            )
            metrics_per_dataset[dataset] = metrics
            results_paths.append(results_path)
        return metrics_per_dataset, results_paths

    def _track_metric(self,
                      dataset: str,
                      metric_score: dict[str, float]) -> str:
        """
        Stores results for given dataset in the corresponding json file
        :param dataset: evaluated dataset
        :param metric_score: dictionary with metrics
        :return: path of file that results where stored
        """
        path = os.path.join(self.results_dir, dataset + '.json')
        with open(path, 'w') as results_file:
            json.dump(metric_score, results_file)
        return path

    def _rename_metrics(
        self,
        metric_score: dict[str, float]
    ) -> dict[str, float]:
        """
        Rename the metrics to remove '@' from the metric names
        :param metric_score: dictionary with metrics
        :return: dictionary with renamed metrics"""
        renamed_metric = {}
        for metric, score in metric_score.items():
            renamed_metric[metric.replace('@', '_')] = score
        return renamed_metric

    def _concat_metrics(
        self,
        ndcg: dict[str, float],
        recall: dict[str, float],
        _map: dict[str, float],
        precision: dict[str, float]
    ) -> dict[str, float]:
        """
        Concatenate the metrics into a single dictionary
        :param ndcg: dictionary with ndcg metrics
        :param recall: dictionary with recall metrics
        :param _map: dictionary with map metrics
        :param precision: dictionary with precision metrics
        :return: dictionary with all metrics"""
        flatten_metrics = {}
        for metric in (ndcg, recall, _map, precision):
            flatten_metrics.update(metric)
        return flatten_metrics

    def _eval_pipeline(
        self,
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        dataset: str
    ) -> tuple[dict[str, float], str]:
        """
        Evaluation of the results of a pipeline and log them in MLFlow
        :param qrels: the relevance of each query-doc pair
        :param results: the results of the pipeline
        :param dataset: the dataset name
        :return: dictionary of metrics
        """
        ndcg, _map, recall, precision = self.retriever.evaluate(
            qrels=qrels,
            results=results,
            k_values=self.retriever.k_values
        )
        ndcg = self._rename_metrics(metric_score=ndcg)
        _map = self._rename_metrics(metric_score=_map)
        recall = self._rename_metrics(metric_score=recall)
        precision = self._rename_metrics(metric_score=precision)
        flatten_metrics = self._concat_metrics(ndcg=ndcg,
                                               recall=recall,
                                               _map=_map,
                                               precision=precision)
        results_path = self._track_metric(
            dataset=dataset,
            metric_score=flatten_metrics
        )
        print('Results for', dataset)
        print('NDCG:', ndcg)
        print("Recall:", recall)
        print('Precision:', precision)
        print('MAP:', _map)
        return flatten_metrics, results_path


class RerankExperiment(Experiment):
    def __init__(self,
                 datasets: list[str],
                 datasets_path: str,
                 onnx_model: CustomModel,
                 batch_size: int,
                 top_k: int,
                 score_function: str,
                 es_hostname: str,
                 initialize: bool,
                 run_name: str):
        self.k = top_k
        self.es_hostname = es_hostname
        self.initialize = initialize
        super().__init__(datasets, datasets_path, onnx_model, batch_size, score_function, run_name)

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
                 datasets: list[str],
                 datasets_path: str,
                 onnx_model: CustomModel,
                 ce_model: str,
                 bi_batch_size: int,
                 ce_batch_size: int,
                 top_k: int,
                 score_function: str,
                 es_hostname: str,
                 initialize: bool,
                 run_name: str):
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
        :param run_name: the name of the given run
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
                         run_name=run_name)

    def _rerank_pipeline(self,
                          corpus: dict[str, dict[str, str]],
                          queries: dict[str, str],
                          index_name: str) \
            -> dict[str, dict[str, float]]:
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
                 datasets: list[str],
                 datasets_path: str,
                 onnx_model: CustomModel,
                 ce_model: str,
                 bi_batch_size: int,
                 ce_batch_size: int,
                 top_k: int,
                 score_function: str,
                 es_hostname: str,
                 initialize: bool,
                 run_name: str):
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
        :param run_name: the name of the given run
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
                         run_name=run_name)

    def _rerank_pipeline(self,
                         corpus: dict[str, dict[str, str]],
                         queries: dict[str, str],
                         index_name: str) -> dict[str, dict[str, float]]:
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
