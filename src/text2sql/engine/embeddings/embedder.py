from abc import ABC, abstractmethod

import tqdm

from tenacity import retry, wait_random_exponential, stop_after_attempt

from text2sql.engine.clients import *


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, samples: list[str]) -> list[list[float]]:
        pass


class AzureEmbedder(BaseEmbedder):
    def __init__(
            self, 
            api_key: str, 
            api_version: str, 
            azure_endpoint: str, 
            model: str, 
            batch_size: int=8, 
            max_chars: int=1024,
            verbose: bool=True
        ):
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.model = model
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.client: AzureOpenAI = get_azure_client(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _embed_batch(self, batch_samples: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=batch_samples,
            model=self.model,
        )
        return [list(x.embedding) for x in response.data]
    
    def embed(self, samples: list[str], verbose: bool=False) -> list[list[float]]:
        embeddings: list[list[float]] = []
        iter_list = range(0, len(samples), self.batch_size)
        if verbose:
            iter_list = tqdm.tqdm(iter_list)
        for i in iter_list:
            batch_inputs = [text[:self.max_chars] for text in samples[i:i+self.batch_size]]
            batch_embeddings = self._embed_batch(batch_inputs)
            embeddings.extend(batch_embeddings)
        return embeddings