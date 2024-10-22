import boto3
import cohere
import weaviate

from loguru import logger
from openai import AsyncAzureOpenAI, AzureOpenAI


def get_azure_client(api_key: str, api_version: str, azure_endpoint: str) -> AzureOpenAI:
    """get azure client"""
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )
    return client


def get_async_azure_client(api_key: str, api_version: str, azure_endpoint: str) -> AsyncAzureOpenAI:
    """get async azure client"""
    client = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )
    return client


def get_bedrock_client(region_name: str, service_name: str = "bedrock-runtime") -> boto3.client:
    """get bedrock client"""
    return boto3.client(
        service_name=service_name,
        region_name=region_name,
    )


def get_cohere_client(api_key: str) -> cohere.Client:
    """get cohere client"""
    return cohere.Client(api_key=api_key)


def get_async_cohere_client(api_key: str) -> cohere.AsyncClient:
    """get async cohere client"""
    return cohere.AsyncClient(api_key=api_key)


def get_weaviate_client(host: str, port: int, grpc_port: int) -> weaviate.WeaviateClient:
    """get weaviate client"""
    client = weaviate.connect_to_local(
        host=host,
        port=port,
        grpc_port=grpc_port,
    )
    if not client.is_ready():
        logger.error("weaviate client not ready")
        raise ConnectionError("weaviate client not ready")
    return client
