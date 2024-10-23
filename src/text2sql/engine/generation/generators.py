from abc import ABC, abstractmethod

import json
import tqdm

from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from text2sql.engine.clients import get_azure_client, get_bedrock_client
from text2sql.engine.generation.converters import convert_messages_to_bedrock_format


class AzureGenerator:

    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        model: str,
        **kwargs,
    ):
        """generate text using Azure OpenAI API

        Args:
            api_key (str): azure api key
            api_version (str): azure api version (mm-dd-yyyy)
            azure_endpoint (str): azure endpoint url
            model (str): azure model deployment name
            kwargs: additional azure client specific arguments

        """
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.model = model
        self.client: AzureOpenAI = get_azure_client(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            **kwargs
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict], **kwargs) -> list[list[float]]:
        """embed one batch of texts with azure"""
        chat_completion = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return chat_completion.choices[0].message.content



class BedrockGenerator:
    def __init__(
        self,
        region_name: str,
        model: str,
        **kwargs,
    ):
        """generate text using Bedrock API

        Args:
            region_name (str): bedrock region name
            model (str): bedrock model name
            kwargs: additional azure client specific arguments
        """
        self.region_name = region_name
        self.model = model
        self.client = get_bedrock_client(region_name=self.region_name, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict], **kwargs) -> str:
        system_message, formatted_messages = convert_messages_to_bedrock_format(model=self.model, messages=messages)
        if system_message:
            response = self.client.converse(
                modelId=self.model,
                system=system_message,
                messages=formatted_messages,
                **kwargs,
            )
        else:
            response = self.client.converse(
                modelId=self.model,
                messages=formatted_messages,
                **kwargs,
            )
        return response["output"]["message"]["content"][0]["text"]
        
