import json

from abc import ABC, abstractmethod
from typing import Callable

import tqdm

from together import Together
from openai import AzureOpenAI, OpenAI
import google.generativeai as genai
from tenacity import retry, wait_random_exponential, stop_after_attempt

from text2sql.engine.clients import get_azure_client, get_bedrock_client, get_openai_client, get_togetherai_client
from text2sql.engine.generation.converters import convert_messages_to_bedrock_format


def identity(x: str) -> str:
    return x


class AzureGenerator:

    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        model: str,
        post_func: Callable[[str], str] = identity,
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
        self.post_func = post_func  
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
        return self.post_func(chat_completion.choices[0].message.content)



class BedrockGenerator:
    def __init__(
        self,
        region_name: str,
        model: str,
        post_func: Callable[[str], str] = identity,
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
        self.post_func = post_func  
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
        return self.post_func(response["output"]["message"]["content"][-1]["text"])

    
class GCPGenerator:

    def __init__(
        self,
        api_key: str,
        model: str,
        post_func: Callable[[str], str] = identity,
    ):
        """generate text using GCP API

        Args:
            api_key (str): gcp api key
            model (str): gemini model name
            kwargs: additional gemini specific arguments

        """
        self.model = model
        self.post_func = post_func  

        genai.configure(api_key=api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict], **kwargs) -> list[list[float]]:
        system_instruction = "\n".join([message["content"] for message in messages if message["role"] == "system"])
        if system_instruction:
            client = genai.GenerativeModel(self.model, system_instruction=system_instruction, generation_config=kwargs)
        else:
            client = genai.GenerativeModel(self.model, generation_config=kwargs)
        history = []
        for message in messages[:-1]:
            if message["role"] in ["assistant", "user"]:
                if "content" not in message:
                    print(f"{message=}")
                new_message = {"role" : message["role"], "parts": message["content"]}
                history.append(new_message)

        chat = client.start_chat(history=history)

        result = chat.send_message(messages[-1]["content"])
        return self.post_func(result.text)


class OpenAIGenerator:

    def __init__(
        self,
        api_key: str,
        model: str,
        post_func: Callable[[str], str] = identity,
        base_url: str | None = None,
        **kwargs,
    ):
        """generate text using OpenAI client
        can be used for DeepSeek as well

        Args:
            api_key (str): api key for OpenAI or DeepSeek
            model (str): model identifier
            base_url (str): base url for API calls
            kwargs: additional openai client specific arguments

        """
        self.model = model
        self.post_func = post_func  
        self.client: OpenAI = get_openai_client(
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict], **kwargs) -> list[list[float]]:
        """embed one batch of texts with OpenAI client"""
        chat_completion = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return self.post_func(chat_completion.choices[0].message.content)


class TogetherAIGenerator:

    def __init__(
        self,
        api_key: str,
        model: str,
        post_func: Callable[[str], str] = identity,
        **kwargs,
    ):
        """generate text with together.ai

        Args:
            api_key (str): api key for together.ai
            model (str): model identifier
            kwargs: additional openai client specific arguments

        """
        self.model = model
        self.post_func = post_func  
        self.client: Together = get_togetherai_client(
            api_key=api_key,
            **kwargs
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict], **kwargs) -> list[list[float]]:
        """embed one batch of texts with Together AI client"""
        chat_completion = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return self.post_func(chat_completion.choices[0].message.content)
