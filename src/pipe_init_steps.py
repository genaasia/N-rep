import json
import os
from argparse import Namespace

import numpy as np
import yaml
from loguru import logger

from text2sql.data import SchemaManager, SqliteDataset
from text2sql.data.datasets import SCHEMA_FORMATS
from text2sql.engine.generation import AzureGenerator, GCPGenerator
from text2sql.engine.retrieval import LocalRetriever


def prepare_dataset_information(
    test_database_path: str, table_descriptions_path: str | None, column_meaning_dict: dict | None
) -> tuple[SqliteDataset, SchemaManager]:
    """create a database loader and generate the schema descriptions

    Args:
        test_database_path: path to the test databases base directory
        table_descriptions_path: path to the table descriptions json file
    Returns:
        dataset: SqliteDataset
        schema_manager: SchemaManager
    """
    logger.info(f"Loading dataset from {test_database_path}...")
    dataset = SqliteDataset(test_database_path)
    logger.info("Creating schema manager and generating schema descriptions, this may takes some time...")
    schema_manager = SchemaManager(dataset, table_descriptions_path=table_descriptions_path, column_meaning_dict=column_meaning_dict)
    return dataset, schema_manager


def prepare_fewshot_retriever(embeddings_path: str, embeddings_data_path: str) -> LocalRetriever:
    """create an in-memory few-shot similarity retriever and load it with preprocessed vectors and data

    Args:
        embeddings_path: path to the preprocessed numpy embeddings file
        embeddings_data_path: path to the preprocessed json embeddings data file
    Returns:
        retriever: LocalRetriever
    """
    logger.info(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    logger.info(f"Loading embeddings data from {embeddings_data_path}...")
    embeddings_data = json.load(open(embeddings_data_path))
    if len(embeddings) != len(embeddings_data):
        err_message = f"Embeddings and data length mistmatch: {len(embeddings)} != {len(embeddings_data)}"
        logger.error(err_message)
        raise ValueError(err_message)
    retriever = LocalRetriever(embeddings, embeddings_data)
    return retriever


def test_generators():
    """verify generators are working"""
    test_messages = [{"role": "user", "content": "What is the capital of South Korea? Answer in one word."}]
    test_azure_generator = AzureGenerator(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    test_gcp_generator = GCPGenerator(
        model="gemini-1.5-flash",
        api_key=os.getenv("GCP_KEY"),
    )
    p = test_azure_generator.generate(test_messages, temperature=0.0)
    logger.info(f"Azure generator test response: '{p}'")
    p = test_gcp_generator.generate(test_messages, temperature=0.0)
    logger.info(f"Gemini generator test response: '{p}'")


def verify_env_vars():
    """verify environment variables are set"""
    if os.getenv("AZURE_OPENAI_API_KEY") is None:
        raise ValueError("AZURE_OPENAI_API_KEY is not set")
    if os.getenv("AZURE_OPENAI_API_ENDPOINT") is None:
        raise ValueError("AZURE_OPENAI_API_ENDPOINT is not set")
    if os.getenv("AZURE_OPENAI_API_VERSION") is None:
        raise ValueError("AZURE_OPENAI_API_VERSION is not set")
    if os.getenv("AZURE_OPENAI_MODEL") is None:
        raise ValueError("AZURE_OPENAI_MODEL is not set")
    if os.getenv("GCP_KEY") is None:
        raise ValueError("GCP_KEY is not set")
    if os.getenv("AWS_ACCESS_KEY_ID") is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    if os.getenv("AWS_SECRET_ACCESS_KEY") is None:
        raise ValueError("AWS_SECRET_ACCESS_KEY is not set")


def verify_required_files(args: Namespace):
    """verify required files exist"""
    for path in [
        args.test_json_path,
        args.test_tables_json_path,
        args.embeddings_path,
        args.embeddings_data_path,
        args.candidate_configs_path,
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    # validate all required json files are type json
    for path in [
        args.test_json_path,
        args.test_tables_json_path,
        args.embeddings_data_path,
    ]:
        if not path.endswith(".json"):
            raise ValueError(f"Required file is not json: {path}")
    # validate all numpy files exist
    for path in [args.embeddings_path]:
        if not path.endswith(".npy"):
            raise ValueError(f"Required file is not numpy: {path}")

    # validate test_database_path exists and is a directory
    if not os.path.isdir(args.test_database_path):
        raise FileNotFoundError(f"Databases directory not found: {args.test_database_path}")


def load_candidate_configs(candidate_configs_path: str) -> tuple[list[dict], int]:
    """load candidate configs from yaml file"""
    top_k = 3  # 3 by default, can override in candidate configs
    with open(candidate_configs_path, "r") as f:
        candidate_config_data: list[dict] = yaml.safe_load(f)
        if "configs" not in candidate_config_data:
            raise ValueError("candidate_config_data must contain a 'configs' key")
        if "top_k" in candidate_config_data:
            top_k = candidate_config_data["top_k"]
        candidate_configs: list[dict] = candidate_config_data["configs"]
    for config_idx, config in enumerate(candidate_configs):
        logger.debug(f"Candidate config {config_idx}: {json.dumps(config)}")

    # verify candidate config keys:
    for config in candidate_configs:
        assert "schema_format" in config
        assert "schema_filtering" in config
        assert "generator" in config
        assert "model" in config
        assert config["schema_format"] in SCHEMA_FORMATS

    return candidate_configs, top_k


def create_output_dir(output_path, candidate_configs):
    """create output directory and save copy of candidate configs"""
    if not os.path.isdir(output_path):
        logger.info(f"Output directory not found, creating it: {output_path}")
        os.makedirs(output_path)
        # save copy of candidate configs, to confirm against when loading
        with open(os.path.join(output_path, "experiment_candidate_configs.yaml"), "w") as f:
            yaml.dump(candidate_configs, f)
    else:
        logger.info(f"Output directory found, existing outputs will be overwritten: {output_path}")
        # check if candidate configs match
        if not os.path.isfile(os.path.join(output_path, "experiment_candidate_configs.yaml")):
            raise FileNotFoundError("copy of experiment_candidate_configs.yaml not found in output directory")
        with open(os.path.join(output_path, "experiment_candidate_configs.yaml"), "r") as f:
            candidate_configs_copy: list[dict] = yaml.safe_load(f)
            if candidate_configs != candidate_configs_copy:
                raise ValueError("candidate_configs mismatch! must have same configs for restoring data")
