import argparse
import json
import os

import numpy as np
import tqdm

from dotenv import load_dotenv
from loguru import logger

from text2sql.data import SqliteDataset, SchemaManager

from text2sql.engine.embeddings import BaseEmbedder, BedrockCohereEmbedder
from text2sql.engine.generation import BaseGenerator, AzureGenerator, GCPGenerator
from text2sql.engine.retrieval import LocalRetriever


def prepare_dataset_information(
    test_database_path: str, table_descriptions_path: str | None
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
    logger.info(f"Creating schema manager and generating schema descriptions, this may takes some time...")
    schema_manager = SchemaManager(dataset, table_descriptions_path=table_descriptions_path)
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
        logger.error(
            f"Embeddings and embeddings data must have the same length: {len(embeddings)} != {len(embeddings_data)}"
        )
        raise ValueError(
            f"Embeddings and embeddings data must have the same length: {len(embeddings)} != {len(embeddings_data)}"
        )
    retriever = LocalRetriever(embeddings, embeddings_data)
    return retriever


def run_schema_linking(sample: dict, schema_manager: SchemaManager) -> dict:
    """run the schema linking task

    Args:
        sample: one sample from the test set json
        schema_manager: the schema manager
    Returns:
        outputs: dict with the mapping, table_filtered and column_filtered results
    """

    outputs = {}

    return outputs


def run_fewshot_retrieval():
    """run the fewshow retrieval task"""
    return


def run_sql_generation():
    """run the sql generation task"""
    return


def run_candidate_selection():
    """run the candidate selection task"""
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-database-path", type=str, required=True, help="path to the test databases base directory"
    )
    parser.add_argument("--test-json-path", type=str, required=True, help="path to the test.json file")
    parser.add_argument("--test-tables-json-path", type=str, required=True, help="path to the test_tables.json file")
    parser.add_argument(
        "--column-meaning-json-path",
        type=str,
        default=None,
        help="path to the column_meaning.json file, leave blank if not used",
    )
    parser.add_argument("--embeddings-path", type=str, required=True, help="path to preprocessed numpy embeddings file")
    parser.add_argument(
        "--embeddings-data-path", type=str, required=True, help="path to preprocessed json embeddings data file"
    )
    parser.add_argument("--output-path", type=str, required=True, default="../outputs", help="target output path")
    args = parser.parse_args()

    load_dotenv()
    # verify environment variables are set
    logger.info("Validating environment variables...")
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

    logger.info("Validating input files...")
    # validate all required json files exist
    for path in [
        args.test_json_path,
        args.test_tables_json_path,
        args.embeddings_path,
        args.embeddings_data_path,
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

    # if output path does not exist, create it
    if not os.path.isdir(args.output_path):
        logger.info(f"Output directory not found, creating it: {args.output_path}")
        os.makedirs(args.output_path)
    else:
        logger.info(f"Output directory found, existing outputs will be overwritten: {args.output_path}")

    logger.info("Loading test data...")
    # load test.json
    with open(args.test_json_path, "r") as f:
        test_json = json.load(f)

    # load column_meaning.json
    if args.column_meaning_json_path is None:
        with open(args.column_meaning_json_path, "r") as f:
            column_meaning_json = json.load(f)
    else:
        column_meaning_json = {}

    # create generators

    # preprocessing
    dataset, schema_manager = prepare_dataset_information(args.test_database_path, args.test_tables_json_path)
    retriever = prepare_fewshot_retriever(args.embeddings_path, args.embeddings_data_path)

    logger.info("Preprocessing complete, starting inference...")
