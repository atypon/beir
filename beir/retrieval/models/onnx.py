import os
from typing import List, Dict, Union, Tuple
from transformers import AutoTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
import numpy as np

class OnnxBERT:
    def __init__(self, onnx_filename: Union[str, Tuple], model_path: Union[str, Tuple] = None, sep: str = " ", **kwargs):
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

    def encode_queries(self, queries: List[str]) -> Union[
        List[List[float]], np.ndarray, List[float]]:
        inputs = self.__create_ort_input(queries=queries)
        return self.q_model.run([], inputs)[0]

    def encode_corpus(self, corpus: List[Dict[str, str]]) -> Union[
        List[List[float]], np.ndarray, List[float]]:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc
                     in corpus]
        inputs = self.__create_ort_input(queries=sentences)
        return self.doc_model.run([], inputs)[0]

    def __tokenize_text(self, queries: List[str]) -> Tuple[np.ndarray, np.ndarray]:

        encoding = self.tokenizer(queries,
                                  return_token_type_ids=False,
                                  return_tensors='np',
                                  max_length=512,
                                  truncation=True,
                                  padding='max_length')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return input_ids, attention_mask

    def __create_ort_input(self, queries: List[str]) -> Dict[str, List[np.ndarray]]:
        ort_input_ids, ort_attention_mask = self.__tokenize_text(queries=queries)
        inputs = {
            'input_ids': list(ort_input_ids),
            'attention_mask': list(ort_attention_mask)
        }
        return inputs
