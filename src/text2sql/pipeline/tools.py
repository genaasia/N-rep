import os

from loguru import logger
from text2sql.engine.generation import (
    AzureGenerator, 
    BedrockGenerator,
    GCPGenerator,
)
from text2sql.engine.generation.postprocessing import (
    extract_first_code_block, 
    extract_sql_from_json,
)
from text2sql.engine.prompts import (
    ESQLCoTPromptFormatter,
    GenaCoTPromptFormatter,
    LegacyFewShotPromptFormatter,
)


def get_generator(generator_name, model, post_func):
    if generator_name == "azure-gpt":
        logger.debug(f"using '{model}'")
        generator = AzureGenerator(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_API_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            model=model,
            post_func=post_func,
        )
    elif generator_name == "gcp-gemini":
        logger.debug(f"using '{model}'")
        generator = GCPGenerator(
            api_key=os.environ.get("GCP_API_KEY"),
            model=model,
            post_func=post_func,
        )
    elif generator_name == "aws-bedrock":
        generator = BedrockGenerator(
            region_name="us-west-2",
            model=model,
            post_func=post_func,
        )
    else:
        raise Exception(f"No known generator with the name {generator_name}")
    return generator


def get_formatter(formatter_name, database_type):
    if formatter_name == "legacy":
        formatter = LegacyFewShotPromptFormatter(database_type=database_type)
    elif formatter_name == "ESQLCoT":
        formatter = ESQLCoTPromptFormatter(database_type=database_type)
    elif formatter_name == "GENACoT":
        formatter = GenaCoTPromptFormatter(database_type=database_type)
    else:
        raise Exception(f"No known formatter with the name {formatter_name}")
    return formatter


def get_schema_description(db_name, schema_name, db_instance):
    schema_description = db_instance.describe_database_schema(
            db_name, mode=schema_name
        )
    return schema_description


def get_postfunc(postfunc_name):
    if postfunc_name == "extract_first_code_block":
        return extract_first_code_block
    elif postfunc_name == "extract_sql_from_json":
        return extract_sql_from_json
    raise Exception(f"No known postfunc with the name {postfunc_name}")
