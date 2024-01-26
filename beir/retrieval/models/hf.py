from typing import List, Union, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F


class HFModel(object):
	
	def __init__(self,
	             model_path: str,
	             max_seq_length: int,
	             sep: str = " "
	             ) -> None:
		
		self.q_model = AutoModelForSequenceClassification.from_pretrained(
			pretrained_model_name_or_path=model_path,
			trust_remote_code=True,
		)
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.q_model.to(device)
		self.doc_model = self.q_model
		
		self.tokenizer = AutoTokenizer.from_pretrained(
			pretrained_model_name_or_path=model_path,
			model_max_length=max_seq_length,
		)
		
		self.max_seq_length = max_seq_length
		self.sep = sep
	
	@staticmethod
	def _batchify(queries: List[str], batch_size: int) -> List[List[str]]:
		batches = []
		for i in range(0, len(queries), batch_size):
			batches.append(queries[i: i + batch_size])
		return batches
	
	@torch.inference_mode()
	def encode_queries(self, queries: List[str],
	                   batch_size: int,
	                   convert_to_tensor: bool = None,
	                   show_progress_bar: bool = True) -> Union[List[List[float]], np.ndarray, List[float]]:
		
		batchified_queries = self._batchify(queries=queries, batch_size=batch_size)
		query_embeddings = []
		if show_progress_bar:
			batches = tqdm(batchified_queries, total=len(batchified_queries))
		else:
			batches = tqdm(batchified_queries)
		for batch in batches:
			inputs = self.tokenizer(
				batch,
				return_tensors="pt",
				padding="max_length",
				return_token_type_ids=False,
				truncation=True,
				max_length=self.max_seq_length).to(self.q_model.device)
			query_embeddings += list(self.q_model(**inputs)['sentence_embedding']
			                         .detach()
			                         .cpu()
			                         .numpy())
		query_embeddings = np.asarray(query_embeddings)
		return query_embeddings
	
	@torch.inference_mode()
	def encode_corpus(self, corpus: List[Dict[str, str]],
	                  batch_size: int,
	                  show_progress_bar: bool = True,
	                  normalize_embeddings: bool = False,  # necessary for experiments with faiss
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
			inputs = self.tokenizer(
				batch,
				return_tensors="pt",
				padding="max_length",
				return_token_type_ids=False,
				truncation=True,
				max_length=self.max_seq_length).to(self.doc_model.device)
			corpus_embeddings += list(self.doc_model(**inputs)['sentence_embedding']
			                          .detach()
			                          .cpu()
			                          .numpy())
		corpus_embeddings = np.asarray(corpus_embeddings)
		return corpus_embeddings
	
	
	