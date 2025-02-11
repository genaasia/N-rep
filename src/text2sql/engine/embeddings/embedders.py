from abc import ABC, abstractmethod

import json
import time
import tqdm

from loguru import logger
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_random_exponential, stop_after_attempt

from text2sql.engine.clients import get_azure_client, get_bedrock_client, get_cohere_client


class BaseEmbedder(ABC):
    def __init__(self, batch_size: int = 8, max_chars: int = 1024, sleep_ms: int = 0):
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.sleep_ms: int = 0

    @abstractmethod
    def _embed_batch(self, batch_samples: list[str]) -> list[list[float]]:
        """batch embedding function for specific client"""
        pass

    def embed_list(self, samples: list[str], verbose: bool = False) -> list[list[float]]:
        """embed a list of texts, with optional progress bar"""
        embeddings: list[list[float]] = []
        iter_list = range(0, len(samples), self.batch_size)
        if verbose:
            iter_list = tqdm.tqdm(iter_list)
        for i in iter_list:
            batch_inputs = [text[: self.max_chars] for text in samples[i : i + self.batch_size]]
            batch_embeddings = self._embed_batch(batch_inputs)
            embeddings.extend(batch_embeddings)
            if self.sleep_ms:
                time.sleep(self.sleep_ms / 1000)
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        """embed a single text"""
        return self._embed_batch([text])[0]

    def embed(self, data: str | list[str], verbose: bool = False) -> list[float] | list[list[float]]:
        """lazy function to embed either a single text or a list of texts"""
        if type(data) == str:
            return self.embed_text(data)
        else:
            return self.embed_list(data, verbose=verbose)


class AzureEmbedder(BaseEmbedder):

    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        model: str,
        batch_size: int = 8,
        max_chars: int = 1024,
        sleep_ms: int = 0,
        **kwargs,
    ):
        """embed texts using Azure OpenAI API

        Args:
            api_key (str): azure api key
            api_version (str): azure api version (mm-dd-yyyy)
            azure_endpoint (str): azure endpoint url
            model (str): azure model deployment name
            batch_size (int, optional): batch size. Defaults to 8.
            max_chars (int, optional): max chars. Defaults to 1024.
            sleep_ms (int, optional): sleep time in ms. Defaults to 0.
            kwargs: additional azure client specific arguments
        """
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.model = model
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.sleep_ms = sleep_ms
        self.client: AzureOpenAI = get_azure_client(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            **kwargs,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _embed_batch(self, batch_samples: list[str]) -> list[list[float]]:
        """embed one batch of texts with azure"""
        response = self.client.embeddings.create(
            input=batch_samples,
            model=self.model,
        )
        return [list(x.embedding) for x in response.data]


class BedrockCohereEmbedder(BaseEmbedder):

    def __init__(
        self,
        region_name: str,
        model: str,
        input_type: str,
        embedding_type: str = "float",
        service_name: str = "bedrock-runtime",
        batch_size: int = 8,
        max_chars: int = 1024,
        sleep_ms: int = 0,
    ):
        """embed texts using Cohere embeddings on Amazon Bedrock API

        Args:
            region_name (str): aws region name
            model (str): bedrock model id
            input_type (str): input type
            embedding_type (str, optional): embedding type. Defaults to "float".
            service_name (str, optional): bedrock service name. Defaults to "bedrock-runtime".
            batch_size (int, optional): batch size. Defaults to 8.
            max_chars (int, optional): max chars. Defaults to 1024.
            sleep_ms (int, optional): sleep time in ms. Defaults to 0.
        """
        self.region_name = region_name
        self.service_name = service_name
        self.model = model
        self.input_type = input_type
        self.embedding_type = embedding_type
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.client = get_bedrock_client(
            service_name=self.service_name,
            region_name=self.region_name,
        )
        self.batch_size = batch_size
        self.sleep_ms = sleep_ms

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _embed_batch(self, batch_samples: list[str]) -> list[list[float]]:
        """embed one batch of texts"""
        request_body = json.dumps(
            {
                "texts": batch_samples,
                "input_type": self.input_type,
                "embedding_types": [self.embedding_type],
            }
        )
        response = self.client.invoke_model(
            body=request_body,
            modelId=self.model,
            accept="*/*",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        embeddings_dict: dict = response_body.get("embeddings")
        embeddings = embeddings_dict.get(self.embedding_type)
        return embeddings


class BedrockTitanv2Embedder(BaseEmbedder):

    def __init__(
        self,
        region_name: str,
        model: str,
        dimensions: int = 1024,
        normalize: bool = True,
        embedding_type: str = "float",
        service_name: str = "bedrock-runtime",
        batch_size: int = 1,
        max_chars: int = 1024,
        sleep_ms: int = 0,
    ):
        """embed texts using Titan v2 embeddings on Amazon Bedrock API

        Args:
            region_name (str): aws region name
            model (str): bedrock model id
            dimensions (int): number of dimensions. Defaults to 1024.
            normalize (bool, optional): normalize embeddings. Defaults to True.
            embedding_type (str, optional): embedding type. Defaults to "float".
            service_name (str, optional): bedrock service name. Defaults to "bedrock-runtime".
            batch_size (int, optional): batch size - must be 1 for titan. Defaults to 1.
            max_chars (int, optional): max chars. Defaults to 1024.
            sleep_ms (int, optional): sleep time in ms. Defaults to 0.
        """
        if batch_size != 1:
            logger.warning("batch_size is set to 1 for Titan v2 embeddings")
            batch_size = 1
        self.region_name = region_name
        self.service_name = service_name
        self.model = model
        self.dimensions = dimensions
        self.embedding_type = embedding_type
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.client = get_bedrock_client(
            service_name=self.service_name,
            region_name=self.region_name,
        )
        self.sleep_ms = sleep_ms

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _embed_batch(self, batch_samples: list[str]) -> list[list[float]]:
        """embed one batch of texts"""
        if len(batch_samples) != 1:
            raise ValueError("batch_size must be 1 for Titan v2 embeddings")
        request_body = json.dumps(
            {
                "inputText": batch_samples[0],
                "dimensions": self.dimensions,
                "normalize": self.normalize,
                "embeddingTypes": [self.embedding_type],
            }
        )
        response = self.client.invoke_model(
            body=request_body,
            modelId=self.model,
            accept="*/*",
            contentType="application/json",
        )
        # the documentation uses like response['embeddingByTypes']['binary']
        # except this is not correct!
        response_body = json.loads(response.get("body").read())
        embedding: list = response_body.get("embedding")
        return [embedding]


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(
        self,
        model_path: str,
        cache_dir: str | None = None,
        batch_size: int = 1,
        max_chars: int = 1024,
        sleep_ms: int = 0,
    ):
        """embed texts using sentence-transformers

        Args:
            model_path (str): path to sentence-transformers model (local or huggingface hub)
            cache_dir (str, optional): cache directory for model. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 1.
            max_chars (int, optional): max chars. Defaults to 1024.
            sleep_ms (int, optional): sleep time in ms. Defaults to 0.
        """
        if batch_size != 1:
            logger.warning("batch_size is set to 1 for Titan v2 embeddings")
            batch_size = 1
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.sleep_ms = sleep_ms
        self.model = self._initialize_model(
            model_path=self.model_path,
            cache_dir=self.cache_dir,
        )

    def _initialize_model(self, model_path: str, cache_dir: str | None) -> SentenceTransformer:
        """initialize SentenceTransformer model"""
        if cache_dir:
            model = SentenceTransformer(model_path, cache_folder=cache_dir)
        else:
            model = SentenceTransformer(model_path)
        return model

    def _embed_batch(self, batch_samples: list[str]) -> list[list[float]]:
        """embed one batch of texts"""
        embeddings = self.model.encode(batch_samples)
        return embeddings