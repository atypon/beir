import os
from typing import List
from onnxruntime import InferenceSession
from beir.retrieval.models import OnnxBERT
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

class Experiment(object):
    def __init__(self, datasets: List[str],
                 datasets_path: str,
                 onnx_model: OnnxBERT,
                 batch_size: int,
                 score_function: str):
        self.dataset_paths = []
        for dataset in datasets:
            dataset_folder = os.path.join(datasets_path, dataset)
            self.dataset_paths.append(dataset_folder)
        self.onnx_model = onnx_model
        self.bs = batch_size
        self.score_func = score_function
        self.__setup_experiment()

    def __setup_experiment(self):
        self.model = DRES(self.onnx_model, batch_size=self.bs)
        self.retriever = EvaluateRetrieval(self.model, score_function=self.score_func)

    def experiemnt_pipeline(self):
        for dataset in self.dataset_paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=dataset).load(split='test')
            results = self.retriever.retrieve(corpus=corpus, queries=queries)
            ndcg, _map, recall, precision = self.retriever.evaluate(qrels=qrels,
                                                                    results=results,
                                                                    k_values=self.retriever.k_values)
            print('Results for', dataset)
            print('NDCG:', ndcg)
            print("Recall:", recall)
            print('Precision:', precision)
            print('MAP:', _map)
