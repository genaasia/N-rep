import time

from abc import ABC, abstractmethod
from typing import Callable

import google.generativeai as genai

from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt

from text2sql.engine.clients import get_azure_client, get_bedrock_client, get_openai_client
from text2sql.engine.generation.converters import convert_messages_to_bedrock_format


def identity(x: str) -> str:
    return x


STATUS_OK = "ok"


class TokenUsage(BaseModel):
    """token usage for a single generation call"""

    cached_tokens: int = 0
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    inf_time_ms: int

    # allow adding two TokenUsages
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            cached_tokens=self.cached_tokens + other.cached_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            inf_time_ms=self.inf_time_ms + other.inf_time_ms,
        )


class GenerationResult(BaseModel):
    """result of a single generation call"""

    model: str
    text: str
    tokens: TokenUsage
    status: str = STATUS_OK


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        pass


class AzureGenerator(BaseGenerator):

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
            api_key=self.api_key, api_version=self.api_version, azure_endpoint=self.azure_endpoint, **kwargs
        )

    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(8))
    def generate(self, messages: list[dict], **kwargs) -> GenerationResult:
        """embed one batch of texts with azure"""

        # run inference
        start_time = time.time()
        chat_completion = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        end_time = time.time()

        # get token usage
        if hasattr(chat_completion.usage, "prompt_tokens_details") and hasattr(
            chat_completion.usage.prompt_tokens_details, "cached_tokens"
        ):
            cached_tokens = chat_completion.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0

        # postprocessing
        text = self.post_func(chat_completion.choices[0].message.content)
        inf_time_ms = int((end_time - start_time) * 1000)
        token_usage = TokenUsage(
            cached_tokens=cached_tokens,
            prompt_tokens=chat_completion.usage.prompt_tokens,
            output_tokens=chat_completion.usage.completion_tokens,
            total_tokens=chat_completion.usage.total_tokens,
            inf_time_ms=inf_time_ms,
        )
        return GenerationResult(model=self.model, text=text, tokens=token_usage)


class BedrockGenerator(BaseGenerator):
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

    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(8))
    def generate(self, messages: list[dict], **kwargs) -> GenerationResult:
        # format to nested bedrock format & run inference
        system_message, formatted_messages = convert_messages_to_bedrock_format(model=self.model, messages=messages)
        start_time = time.time()
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
        end_time = time.time()

        # get token usage
        cached_tokens = response["usage"]["cachedReadInputTokens"]
        if cached_tokens is None:
            cached_tokens = 0
        token_usage = TokenUsage(
            cached_tokens=cached_tokens,
            prompt_tokens=response["usage"]["inputTokens"],
            output_tokens=response["usage"]["outputTokens"],
            total_tokens=response["usage"]["totalTokens"],
            inf_time_ms=int((end_time - start_time) * 1000),
        )

        return GenerationResult(
            model=self.model,
            text=self.post_func(response["output"]["message"]["content"][-1]["text"]),
            tokens=token_usage,
        )


class GCPGenerator(BaseGenerator):

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

    @retry(wait=wait_random_exponential(min=3, max=30), stop=stop_after_attempt(3))
    def generate(self, messages: list[dict], **kwargs) -> GenerationResult:
        # create client depending on system prompt
        system_instruction = "\n".join([message["content"] for message in messages if message["role"] == "system"])
        if system_instruction:
            client = genai.GenerativeModel(self.model, system_instruction=system_instruction, generation_config=kwargs)
        else:
            client = genai.GenerativeModel(self.model, generation_config=kwargs)
        # format messages to GCP format
        history = []
        for message in messages[:-1]:
            if message["role"] in ["assistant", "user"]:
                if "content" not in message:
                    print(f"{message=}")
                new_message = {"role": message["role"], "parts": message["content"]}
                history.append(new_message)

        # run inference
        start_time = time.time()
        chat = client.start_chat(history=history)
        result = chat.send_message(messages[-1]["content"])
        end_time = time.time()

        # get token usage
        if hasattr(result, "usage_metadata"):
            cached_tokens = result.usage_metadata.candidates_token_count
            prompt_tokens = result.usage_metadata.prompt_token_count
            output_tokens = result.usage_metadata.candidates_token_count
            total_tokens = result.usage_metadata.total_token_count
            status = STATUS_OK
        else:
            cached_tokens = 0
            prompt_tokens = 0
            output_tokens = 0
            total_tokens = 0
            status = "error: no usage metadata"
        token_usage = TokenUsage(
            cached_tokens=cached_tokens,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            inf_time_ms=int((end_time - start_time) * 1000),
        )
        return GenerationResult(
            model=self.model,
            text=self.post_func(result.text),
            tokens=token_usage,
            status=status,
        )


class OpenAIGenerator(BaseGenerator):

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
        self.client: OpenAI = get_openai_client(api_key=api_key, base_url=base_url, **kwargs)

    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(8))
    def generate(self, messages: list[dict], **kwargs) -> GenerationResult:
        """embed one batch of texts with azure"""

        # run inference
        start_time = time.time()
        chat_completion = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        end_time = time.time()

        # get token usage
        if hasattr(chat_completion.usage, "prompt_tokens_details") and hasattr(
            chat_completion.usage.prompt_tokens_details, "cached_tokens"
        ):
            cached_tokens = chat_completion.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0

        # postprocessing
        text = self.post_func(chat_completion.choices[0].message.content)

        token_usage = TokenUsage(
            cached_tokens=cached_tokens,
            prompt_tokens=chat_completion.usage.prompt_tokens,
            output_tokens=chat_completion.usage.completion_tokens,
            total_tokens=chat_completion.usage.total_tokens,
            inf_time_ms=int((end_time - start_time) * 1000),
        )

        return GenerationResult(model=self.model, text=text, tokens=token_usage)
