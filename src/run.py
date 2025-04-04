import argparse
import json
import os

from typing import Literal

import numpy as np
import tqdm

from dotenv import load_dotenv
from loguru import logger

from bird_data.schema_linking_data import SCHEMA_LINKING_EXAMPLES

from text2sql.data import BaseDataset, SqliteDataset, SchemaManager
from text2sql.engine.embeddings import BaseEmbedder, BedrockCohereEmbedder
from text2sql.engine.generation import BaseGenerator, AzureGenerator, GCPGenerator
from text2sql.engine.prompts.formatters import GenaCoTwEvidencePromptFormatter
from text2sql.engine.prompts.formatters import SchemaLinkingFewShotFormatter
from text2sql.engine.retrieval import LocalRetriever
from text2sql.engine.generation.postprocessing import extract_first_code_block
from text2sql.utils import parse_json_from_prediction, replace_entities_with_tokens

from text2sql.pipeline.selection import select_best_candidate


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


def run_schema_linking(
    generator: BaseGenerator,
    schema_manager: SchemaManager,
    sample: dict,
    schema_format: str,
) -> dict:
    """run the schema linking task

    Args:
        generator: LLM generator
        schema_manager: the schema manager
        sample: one sample from the test set json
        schema_format: schema mode to use
    Returns:
        outputs: dict with the table_linking, column_linking, table_description and column_description
    """
    db_id = sample["db_id"]
    question = sample["question"]
    evidence = sample.get("evidence", "")

    # generate the input messages and run inference
    message_formatter = SchemaLinkingFewShotFormatter(SCHEMA_LINKING_EXAMPLES, description_format=schema_format)
    schema_description = schema_manager.get_full_schema(db_id, schema_format)
    messages = message_formatter.generate_messages(schema_description, question, evidence)
    raw_prediction = generator.generate(messages, temperature=0.0)
    try:
        full_linking: dict = parse_json_from_prediction(raw_prediction)
        table_linking = dict([(table, "keep_all") for table in full_linking.keys()])
        column_description = schema_manager.get_filtered_schema(db_id, full_linking, schema_format)
        table_description = schema_manager.get_filtered_schema(db_id, table_linking, schema_format)
    except Exception as e:
        logger.error(f"Error parsing schema linking prediction, returning all: {str(e)}")
        full_linking = None
        table_linking = None
        column_description = schema_description
        table_description = schema_description

    outputs = {
        "messages": messages,
        "prediction": raw_prediction,
        "table_linking": table_linking,
        "column_linking": full_linking,
        "table_description": table_description,
        "column_description": column_description,
        "full_description": schema_description,
    }

    return outputs


def run_fewshot_retrieval(
    embedder: BaseEmbedder,
    retriever: LocalRetriever,
    sample: dict,
    top_k: int = 3,
    do_mask: bool = True,
) -> list[dict]:
    """run the fewshow retrieval task

    Args:
        embedder: embedder
        retriever: retriever pre-loaded with vectors and data
        sample: one sample from the test set json
        top_k: number of top k results to return
    Returns:replace_entities_with_tokens
        results: list of dict with the top k results (with keys id, distance, data)
    """
    question = sample["question"]
    if do_mask:
        question = replace_entities_with_tokens(question)
    embedding = embedder.embed(question)
    results = retriever.query(embedding, top_k=top_k)
    return results


def run_sql_generation(
    generator: BaseGenerator,
    sample: dict,
    few_shot_results: list[dict],
    schema_linking_outputs: dict,
    schema_format: str,
    schema_filtering: Literal["none", "table", "column"],
):
    """run the sql generation task"""
    # format messages
    few_shot_examples: list[dict] = [result for result in few_shot_results if schema_format in result["data"]]
    message_formatter = GenaCoTwEvidencePromptFormatter(
        database_type="sqlite",
        few_shot_query_key="nl_en_query",
        few_shot_target_key="sql_query",
        fewshot_schema_key=schema_format,
    )
    if schema_filtering == "table":
        schema_description = schema_linking_outputs["table_description"]
    elif schema_filtering == "column":
        schema_description = schema_linking_outputs["column_description"]
    else:
        schema_description = schema_linking_outputs["full_description"]
    messages = message_formatter.generate_messages(
        schema_description=schema_description,
        query=sample["question"],
        evidence=sample.get("evidence", ""),
        few_shot_examples=few_shot_examples,
    )
    # run inference
    raw_prediction = generator.generate(messages, temperature=0.0)
    sql_prediction = extract_first_code_block(raw_prediction)
    if not sql_prediction:
        sql_prediction = raw_prediction
    return sql_prediction


def run_candidate_selection(
        dataset: BaseDataset, 
        schema_manager: SchemaManager, 
        generator: BaseGenerator,
        sample: dict, 
        candidate_sqls: list[str],
        chase: bool = False,
    ) -> str:
    """run the candidate selection task"""
    # for each sample, prepare the execution result dict
    database: str = sample["db_id"]
    sample_dicts: list[dict] = []
    for sql_query in candidate_sqls:
        try:
            execution_results: list[dict] = dataset.query_database(database, sql_query)
            is_valid = True
        except Exception as e:
            execution_results = []
            is_valid = False

        sample_dicts.append({
            "sql": sql_query,
            "valid": is_valid,
            "results": execution_results,
        })
    # run selection
    best_sql = select_best_candidate(
        predictions=sample_dicts,
        schema_manager=schema_manager,
        db_id=database,
        question=sample["question"],
        evidence=sample.get("evidence", ""),
        generator=generator,
        chase=chase,
    )
    return best_sql


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
        test_data: list[dict] = json.load(f)

    # load column_meaning.json
    if args.column_meaning_json_path is not None:
        with open(args.column_meaning_json_path, "r") as f:
            column_meaning_json = json.load(f)
    else:
        column_meaning_json = {}

    # create generators
    logger.info("Creating embedder...")
    embedder = BedrockCohereEmbedder(
        model=os.getenv("AWS_MODEL_NAME"),
        region_name=os.getenv("AWS_REGION_NAME"),
        input_type=os.getenv("AWS_INPUT_TYPE"),
    )

    logger.info("Creating & testing azure generator...")
    test_messages = [{"role": "user", "content": "What is the capital of South Korea? Answer in one word."}]

    azure_generator = AzureGenerator(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    p = azure_generator.generate(test_messages, temperature=0.0)
    logger.info(f"Azure generator test response: {p}")

    logger.info("Creating & testing Gemini generator...")
    gcp_generator = GCPGenerator(
        model="gemini-1.5-flash",
        api_key=os.getenv("GCP_KEY"),
    )

    p = gcp_generator.generate(test_messages, temperature=0.0)
    logger.info(f"Gemini generator test response: {p}")

    # preprocessing
    dataset, schema_manager = prepare_dataset_information(args.test_database_path, args.test_tables_json_path)
    retriever = prepare_fewshot_retriever(args.embeddings_path, args.embeddings_data_path)

    logger.info("Preprocessing complete, starting inference...")

    # example single inference
    logger.warning("!!!!! Running single test !!!!!")

    sample = test_data[0]
    top_k = 3

    candidate_sqls: list[str] = []
    for schema_format, schema_filtering_mode in [
        ("sql", "none"),
        ("m_schema", "column"),
        ("mac_schema", "table"),
    ]:
        logger.info(f"[test] doing schema linking with '{schema_format}' schema format and '{schema_filtering_mode}' schema filtering")
        schema_linking_outputs = run_schema_linking(gcp_generator, schema_manager, sample, schema_format)
        logger.info(
            f"[test] schema linking output: type {type(schema_linking_outputs).__name__} with keys {schema_linking_outputs.keys()}"
        )
        logger.info("[test] doing schema linking")
        few_shot_results = run_fewshot_retrieval(embedder=embedder, retriever=retriever, sample=sample, top_k=top_k)
        logger.info(
            f"[test] fewshot retrieval output: type {type(few_shot_results).__name__} with keys {few_shot_results[0].keys()}"
        )
        logger.info(f"[test] doing sql generation with gcp generator, '{schema_filtering_mode}' schema filtering")
        sql = run_sql_generation(
            generator=gcp_generator,
            sample=sample,
            few_shot_results=few_shot_results,
            schema_linking_outputs=schema_linking_outputs,
            schema_format=schema_format,
            schema_filtering=schema_filtering_mode,
        )
        candidate_sqls.append(sql)
        logger.info(f"[test] predicted sql candidate: {sql}")

    best_sql = run_candidate_selection(
        dataset=dataset,
        schema_manager=schema_manager,
        sample=sample,
        candidate_sqls=candidate_sqls,
        generator=gcp_generator,
        chase=True,
    )
    logger.info(f"[test] best sql: {best_sql}")
        


if __name__ == "__main__":
    main()
