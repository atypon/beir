import os
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
import numpy as np


class OnnxBERT:
    def _init_(self,
               onnx_filename: Union[str, Tuple],
               model_path: Union[str, Tuple] = None,
               sep: str = " ",
               **kwargs):
        self.sep = sep
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        session_options = SessionOptions()
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        if isinstance(model_path, str) and isinstance(onnx_filename, str):
            self.q_model = InferenceSession(os.path.join(model_path, onnx_filename),
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                            sess_options=session_options)
            self.doc_model = self.q_model
        elif isinstance(model_path, tuple) and isinstance(onnx_filename, tuple):
            self.q_model = InferenceSession(os.path.join(model_path[0], onnx_filename[0]),
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                            sess_options=session_options)
            self.doc_model = InferenceSession(os.path.join(model_path[1], onnx_filename[1]),
                                              providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                              sess_options=session_options)

    def encode_queries(self, queries: List[str],
                       batch_size: int,
                       convert_to_tensor: bool = None,
                       show_progress_bar: bool = True) -> Union[
        List[List[float]], np.ndarray, List[float]]:
        batchified_queries = self._batchify(queries=queries, batch_size=batch_size)
        query_embeddings = []
        if show_progress_bar:
            batches = tqdm(batchified_queries, total=len(batchified_queries))
        else:
            batches = tqdm(batchified_queries)
        for batch in batches:
            inputs = self._create_ort_input(queries=batch)
            query_embeddings += list(self.q_model.run([], inputs)[0])
        query_embeddings = np.asarray(query_embeddings)
        return query_embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]],
                      batch_size: int,
                      show_progress_bar: bool = True,
                      normalize_embeddings: bool = False, # necessary for experiments with faiss
                      convert_to_tensor: bool = None) -> Union[
        List[List[float]], np.ndarray, List[float]]:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc
                     in corpus]
        batchified_sentences = self._batchify(queries=sentences, batch_size=batch_size)
        corpus_embeddings = []
        if show_progress_bar:
            batches = tqdm(batchified_sentences, total=len(batchified_sentences))
        else:
            batches = tqdm(batchified_sentences)
        for batch in batches:
            inputs = self._create_ort_input(queries=batch)
            corpus_embeddings += list(self.doc_model.run([], inputs)[0])
        corpus_embeddings = np.asarray(corpus_embeddings)
        return corpus_embeddings

    def _tokenize_text(self, queries: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        encoding = self.tokenizer(queries,
                                  return_token_type_ids=False,
                                  return_tensors='np',
                                  max_length=256,
                                  truncation=True,
                                  padding='max_length')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return input_ids, attention_mask

    def _create_ort_input(self, queries: List[str]) -> Dict[str, List[np.ndarray]]:
        ort_input_ids, ort_attention_mask = self._tokenize_text(queries=queries)
        inputs = {
            'input_ids': list(ort_input_ids),
            'attention_mask': list(ort_attention_mask)
        }
        return inputs

    @staticmethod
    def _batchify(queries: List[str], batch_size: int) -> List[List[str]]:
        batches = []
        for i in range(0, len(queries), batch_size):
            batches.append(queries[i: i+batch_size])
        return batches


class OnnxBGE(OnnxBERT):
    """
    Load BGE model into Inference sessions. The difference with the OnnxBERT class is that in the query BGE prepends
    an instruction at the beginning to let the model extract better embeddings for queries.
    """
    def __init__(self,
                 onnx_filename: Union[str, Tuple],
                 query_instruction: str,
                 model_path: Union[str, Tuple] = None,
                 sep: str = " ",
                 enable_query_instruction: bool = False,
                 **kwargs):
        self.query_instruction = query_instruction
        self.enable_query_instruction = enable_query_instruction
        super().__init__(onnx_filename=onnx_filename,
                         model_path=model_path,
                         sep=sep,
                         kwargs=kwargs)

    def encode_queries(self,
                       queries: List[str],
                       batch_size: int,
                       convert_to_tensor: bool = None,
                       show_progress_bar: bool = True) -> Union[List[List[float]], np.ndarray, List[float]]:
        if self.enable_query_instruction:
            print("Enabling Query instruction!")
            queries = [self.query_instruction + " " + query for query in queries]
        batchified_queries = self._batchify(queries=queries, batch_size=batch_size)
        query_embeddings = []
        if show_progress_bar:
            batches = tqdm(batchified_queries, total=len(batchified_queries))
        else:
            batches = tqdm(batchified_queries)
        for batch in batches:
            inputs = self._create_ort_input(queries=batch)
            query_embeddings += list(self.q_model.run([], inputs)[0])
        query_embeddings = np.asarray(query_embeddings)
        return query_embeddings
