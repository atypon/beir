import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from beir_extensions.models.base_model import CustomModel


class SentenceTransformersModel(CustomModel):
    """
    A class to handle Sentence Transformers models.
    """

    def __init__(
        self,
        model_name: str,
        matryoshka_dim: int | None,
        query_prompt: str | None = None,
        corpus_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt_name: str | None = None
    ):
        """
        Initialize the SentenceTransformers model.

        :param model_name: The name of the Sentence Transformers model.
        :param matryoshka_dim: The dimension to truncate the model to.
        :param query_prompt: Optional prompt to prepend to each query.
            If None, no prompt is used.
        :param corpus_prompt: Optional prompt to prepend to each document in
            the corpus. If None, no prompt is used.
        :param query_prompt_name: Optional name for the query prompt.
            If None, no name is used. If a prompt is provided, this will be
            ignored.
        :param corpus_prompt_name: Optional name for the corpus prompt.
            If None, no name is used. If a prompt is provided, this will be
            ignored.
        """
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            trust_remote_code=True,
            truncate_dim=matryoshka_dim
        )
        self.query_prompt = query_prompt
        self.corpus_prompt = corpus_prompt
        self.query_prompt_name = query_prompt_name
        self.corpus_prompt_name = corpus_prompt_name

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Encode a list of queries into embeddings.
        :param queries: List of queries to encode.
        :param batch_size: Batch size for encoding.
        :param show_progress_bar: Whether to show a progress bar during
            encoding.
        :return: List of query embeddings.
        """
        query_embeddings = self.model.encode(
            queries,
            prompt=self.query_prompt,
            prompt_name=self.query_prompt_name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )
        return query_embeddings

    def encode_corpus(
        self,
        corpus: list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Encode a corpus into embeddings.
        :param corpus: List of documents in the corpus.
        :param batch_size: Batch size for encoding.

        :param show_progress_bar: Whether to show a progress bar during
            encoding.
        :return: List of corpus embeddings.
        """
        corpus_embeddings = self.model.encode(
            corpus,
            prompt=self.corpus_prompt,
            prompt_name=self.corpus_prompt_name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )
        return corpus_embeddings
