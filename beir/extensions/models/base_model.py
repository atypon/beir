from abc import ABC, abstractmethod

import numpy as np


class CustomModel(ABC):
    """
    Abstract base class for custom models.
    Custom models should inherit from this class and implement the
    encode_queries, encode_corpus methods.
    """

    @abstractmethod
    def encode_queries(
        self,
        queries: list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False
    ) -> list[list[float]] | list[float] | np.ndarray:
        """
        Encode a list of queries into embeddings.

        :param queries: List of queries to encode.
        :param batch_size: Batch size for encoding.
        :param show_progress_bar: Whether to show a progress bar during
            encoding.
        :param convert_to_tensor: Whether to convert the embeddings to a
            tensor format.
        :return: List of query embeddings.
        """
        pass

    @abstractmethod
    def encode_corpus(
        self,
        corpus: list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = False
    ) -> list[list[float]] | list[float] | np.ndarray:
        """
        Encode a corpus into embeddings.

        :param corpus: List of documents in the corpus.
        :param batch_size: Batch size for encoding.
        :param show_progress_bar: Whether to show a progress bar during
            encoding.
        :param convert_to_tensor: Whether to convert the embeddings to a
            tensor format.
        :return: List of corpus embeddings.
        """
        pass
