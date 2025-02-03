import json
import uuid
from abc import ABC, abstractmethod

import numpy as np
import tqdm
import weaviate
import weaviate.classes as wvc

from sklearn.metrics.pairwise import distance_metrics, pairwise_distances
from sklearn.preprocessing import normalize
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.util import generate_uuid5


class BaseRetriever(ABC):

    @abstractmethod
    def query():
        pass


class LocalRetriever(BaseRetriever):
    
    def __init__(self, embeddings: list[list[float]] | np.ndarray, data: list[dict], norm: bool = False, distance_metric: str = "cosine"):
        """vector similarity retrieval for local retrieval
        
        Args:
            embeddings (list[list[float]] | np.ndarray): list of embeddings
            data (list[dict]): list of data
            norm (bool, optional): normalize embeddings. Defaults to False.
            distance_metric (str, optional): distance metric. Defaults to "cosine".
        """
        if len(embeddings) != len(data):
            raise ValueError("The number of embeddings must equal the number of data!")
        if distance_metric not in distance_metrics():
            raise ValueError(f"Unknown distance metric '{distance_metric}', must be one of {list(distance_metrics().keys())}")
        if norm:
            embeddings = normalize(embeddings, norm="l2")
        self.distance_metric = distance_metric
        self.embeddings = np.array(embeddings)
        self.data = data

    def query(self, query_vector: list[float] | np.ndarray, top_k: int = 10, distance_metric: str | None = None) -> list[dict]:
        """query the retriever
        
        Args:
            query_vector (list[float] | np.ndarray): query vector
            top_k (int, optional): number of results. Defaults to 10.
            distance_metric (str | None, optional): override default distance metric. Defaults to None."""
        if not distance_metric:
            distance_metric = self.distance_metric
        elif distance_metric not in distance_metrics():
            raise ValueError(f"Unknown distance metric '{distance_metric}', must be one of {list(distance_metrics().keys())}")
        query_vector = np.array(query_vector).reshape(1, -1)
        distances = pairwise_distances(query_vector, self.embeddings, metric=distance_metric)[0]
        indices = np.argsort(distances)
        results = [{"id": int(i), "distance": float(distances[i]), "data": self.data[i]} for i in indices[:top_k]]
        return results


def weaviate_properties_from_dict(data_sample: dict) -> list[Property]:
    """get properties from a data sample"""
    # we assume that all data samples are non-null and same type across sample
    # TODO: the list logic doesn't work if data can have empty lists
    properties = []
    for key, value in data_sample.items():
        if isinstance(value, str):
            prop_dtype = DataType.TEXT
        elif isinstance(value, uuid.UUID):
            prop_dtype = DataType.UUID
        elif isinstance(value, bool):
            prop_dtype = DataType.BOOL
        elif isinstance(value, int):
            prop_dtype = DataType.INT
        elif isinstance(value, float):
            prop_dtype = DataType.NUMBER
        elif isinstance(value, dict):
            prop_dtype = DataType.OBJECT
        elif isinstance(value, list):
            if isinstance(value[0], str):
                prop_dtype = DataType.TEXT_ARRAY
            elif isinstance(value[0], str):
                prop_dtype = DataType.INT_ARRAY
            elif isinstance(value[0], float):
                prop_dtype = DataType.NUMBER_ARRAY
            elif isinstance(value[0], bool):
                prop_dtype = DataType.BOOL_ARRAY
            elif isinstance(value[0], dict):
                prop_dtype = DataType.OBJECT_ARRAY
            elif isinstance(value[0], uuid.UUID):
                prop_dtype = DataType.UUID
        else:
            raise ValueError(f"Unknown type for {key=} and {value=}")
        properties.append(Property(name=key, data_type=prop_dtype))
    return properties


class WeaviateRetriever(BaseRetriever):
    
    def __init__(self, host: str, port: int, grpc_port: int, collection_name: str):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.client: weaviate.Client = self._get_weaviate_client(host, port, grpc_port)

    def _get_weaviate_client(self, host: str, port: int, grpc_port: int) -> weaviate.Client:
        """get weaviate client"""
        client: weaviate.Client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port,
        )
        if not client.is_ready():
            raise Exception("weaviate client not ready")
        return client

    def _create_weaviate_collection(self, properties: list[Property]) -> dict:
        """create the weaviate collection"""
        self.client.collections.create(
            self.collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            properties=properties,
        )
        return self.get_collection_info()

    def populate_collection(
            self, 
            embeddings: list[list[float]] | np.ndarray, 
            data: list[dict], 
            delete_existing: bool = False, 
            norm: bool = False, 
            verbose: bool = True,
        ) -> dict:
        """add data to the weaviate collection"""
        if len(embeddings) != len(data):
            raise ValueError("The number of embeddings must equal the number of data!")
        if norm:
            embeddings = normalize(np.array(embeddings), norm="l2")
        if delete_existing:
            self.client.collections.delete(self.collection_name)
        if not self.client.collections.exists(self.collection_name):
            properties = weaviate_properties_from_dict(data[0])
            self._create_weaviate_collection(properties)
        collection = self.client.collections.get(self.collection_name)
        with collection.batch.dynamic() as batch:
            if verbose:
                iter_over = tqdm.trange(len(embeddings))
            else:
                iter_over = range(len(embeddings))
            for i in iter_over:
                embedding = list(embeddings[i])
                datum = data[i]
                batch.add_object(
                    uuid=generate_uuid5(datum),
                    properties=datum,
                    vector=embedding
                )
            if len(collection.batch.failed_objects) > 0:
                raise Exception(f"Failed to import {len(collection.batch.failed_objects)} objects")
        return self.get_collection_info()


    def get_collection_info(self) -> dict:
        if not self.client.collections.exists(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist! please do populate_collection() first!")
        collection = self.client.collections.get(self.collection_name)
        properties = collection.config.get().to_dict()
        count = collection.aggregate.over_all(total_count=True).total_count
        return {
            "collection_name": self.collection_name,
            "properties": properties,
            "count": count,
        }

    def query(self, query_vector: list[float] | np.ndarray, top_k: int = 10) -> list[dict]:
        if not self.client.collections.exists(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist! please do populate_collection() first!")
        collection = self.client.collections.get(self.collection_name)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        results = [
            {
                "id": str(obj.uuid),
                "distance": float(obj.metadata.distance),
                "data": dict(obj.properties),
            }
            for obj in response.objects
        ]
        return results
