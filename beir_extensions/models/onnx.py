import numpy as np
import torch
from onnxruntime import InferenceSession
from tqdm import tqdm
from transformers import AutoTokenizer

from beir_extensions.models.base_model import CustomModel


class OnnxModel(CustomModel):
    def __init__(
        self,
        onnx_path: str | tuple[str, str],
        tokenizer_path: str | tuple[str, str],
        matryoshka_dim: int | None = None,
        query_prompt: str | None = None,
        corpus_prompt: str | None = None,
        sep: str = " ",
        cls: bool = False,
        **kwargs
    ):
        """
        Initialize the OnnxModel.
        :param onnx_path: Path to the ONNX model file or a tuple of paths for
            query and document models.
        :param tokenizer_path: Path to the tokenizer file or a tuple of paths
            for query and document tokenizers.
        :param matryoshka_dim: Dimension to truncate the model to,
            if applicable.
        :param query_prompt: Optional prompt to prepend to each query.
            If None, no prompt is used.
        :param corpus_prompt: Optional prompt to prepend to each document in
            the corpus. If None, no prompt is used.
        :param sep: Separator to use between title and text in
            corpus documents.
        :param cls: Whether to return only the CLS token embedding.
        :param kwargs: Additional keyword arguments.
        """
        self.sep = sep
        self.cls = cls
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.matryoshka_dim = matryoshka_dim
        self.query_prompt = query_prompt
        self.corpus_prompt = corpus_prompt
        if isinstance(onnx_path, str):
            self.q_model = InferenceSession(
                onnx_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.doc_model = self.q_model
        elif isinstance(onnx_path, tuple):
            self.q_model = InferenceSession(
                onnx_path[0],
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.doc_model = InferenceSession(
                onnx_path[1],
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Method to encode queries using the ONNX model.
        :param queries: List of queries to encode.
        :param batch_size: Batch size for encoding.
        :param show_progress_bar: Whether to show a progress bar during
            encoding.
        :param convert_to_tensor: Whether to convert the output to a
            PyTorch tensor.
        :return: array of query embeddings.
        """

        # If a prompt is provided, prepend it to each query
        if self.query_prompt is not None:
            queries = [self.query_prompt + " " + query for query in queries]
        # Batchify the queries
        batchified_queries = self._batchify(
            queries=queries, batch_size=batch_size)
        query_embeddings = []
        if show_progress_bar:
            batches = tqdm(batchified_queries, total=len(batchified_queries))
        else:
            batches = batchified_queries
        # Iterate over the batches and encode each batch
        for batch in batches:
            inputs = self._create_ort_input(queries=batch)
            model_out = self.q_model.run([], inputs)[0]
            if self.matryoshka_dim is not None:
                model_out = model_out[..., :self.matryoshka_dim]
            if self.cls:
                batch_q_embs = list(model_out[:, 0, :])
                query_embeddings += batch_q_embs
            else:
                query_embeddings += list(model_out)
        query_embeddings = np.asarray(query_embeddings)
        if convert_to_tensor:
            query_embeddings = torch.tensor(query_embeddings)
        return query_embeddings

    def encode_corpus(
        self,
        corpus: list[dict[str, str]],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Method to encode a corpus using the ONNX model.
        :param corpus: List of documents in the corpus, each document is a
            dictionary with keys "title" and "text".
        :param batch_size: Batch size for encoding.
        :param show_progress_bar: Whether to show a progress bar during
            encoding.
        :param convert_to_tensor: Whether to convert the output to a
            PyTorch tensor.
        :return: array of corpus embeddings.
        """
        corpus = [
            (doc["title"] + self.sep + doc["text"]).strip()
            if "title" in doc else doc["text"].strip() for doc in corpus
        ]
        # If a prompt is provided, prepend it to each document
        if self.corpus_prompt is not None:
            corpus = [self.corpus_prompt + " " + doc for doc in corpus]
        # Batchify the corpus
        batchified_corpus = self._batchify(
            queries=corpus,
            batch_size=batch_size
        )
        corpus_embeddings = []
        if show_progress_bar:
            batches = tqdm(batchified_corpus, total=len(batchified_corpus))
        else:
            batches = batchified_corpus
        for batch in batches:
            inputs = self._create_ort_input(queries=batch)
            model_out = self.q_model.run([], inputs)[0]
            if self.matryoshka_dim is not None:
                model_out = model_out[..., :self.matryoshka_dim]
            if self.cls:
                batch_c_embs = list(model_out[:, 0, :])
                corpus_embeddings += batch_c_embs
            else:
                corpus_embeddings += list(model_out)
        corpus_embeddings = np.asarray(corpus_embeddings)
        if convert_to_tensor:
            corpus_embeddings = torch.tensor(corpus_embeddings)
        return corpus_embeddings

    def _tokenize_text(
        self,
        queries: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Tokenize the input text using the tokenizer.
        :param queries: List of queries to tokenize.
        :return: Tuple of input_ids and attention_mask as numpy arrays.
        """
        encoding = self.tokenizer(
            queries,
            return_token_type_ids=False,
            return_tensors='np',
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return input_ids, attention_mask

    def _create_ort_input(
        self,
        queries: list[str]
    ) -> dict[str, list[np.ndarray]]:
        """
        Create the input dictionary for the ONNX model.
        :param queries: List of queries to encode.
        :return: Dictionary with input_ids and attention_mask.
        """
        ort_input_ids, ort_attention_mask = self._tokenize_text(
            queries=queries
        )
        inputs = {
            'input_ids': list(ort_input_ids),
            'attention_mask': list(ort_attention_mask)
        }
        return inputs

    @staticmethod
    def _batchify(queries: list[str], batch_size: int) -> list[list[str]]:
        """
        Batchify the queries into smaller lists of size batch_size.
        :param queries: List of queries to batchify.
        :param batch_size: Size of each batch.
        :return: List of batches, each batch is a list of queries.
        """
        batches = []
        for i in range(0, len(queries), batch_size):
            batches.append(queries[i: i+batch_size])
        return batches
