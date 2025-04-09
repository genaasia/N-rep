import argparse
import json
import os
import sys
import warnings

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from typing import Literal

import numpy as np
import tqdm
import yaml

from dotenv import load_dotenv
from loguru import logger

from bird_data.schema_linking_data import SCHEMA_LINKING_EXAMPLES

from text2sql.data import BaseDataset, SqliteDataset, SchemaManager
from text2sql.data.schema_to_text import schema_to_datagrip_format
from text2sql.engine.embeddings import BaseEmbedder, BedrockCohereEmbedder
from text2sql.engine.generation import BaseGenerator, AzureGenerator, GCPGenerator
from text2sql.engine.prompts.formatters import GenaCoTwEvidencePromptFormatter
from text2sql.engine.prompts.formatters import SchemaLinkingFewShotFormatter
from text2sql.engine.prompts.formatters import RewritePromptFormatter
from text2sql.engine.retrieval import LocalRetriever
from text2sql.engine.generation.postprocessing import extract_first_code_block
from text2sql.utils.postprocess import get_table_names_from_query
from text2sql.utils import parse_json_from_prediction, replace_entities_with_tokens, CharacterCounter, TokenCounter

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
    schema_linking_generator: str,
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
    is_gemini = schema_linking_generator == "gcp"
    messages = message_formatter.generate_messages(schema_description, question, evidence, gemini=is_gemini)
    raw_prediction = generator.generate(messages, temperature=0.0)
    try:
        full_linking: dict = parse_json_from_prediction(raw_prediction)
        table_linking = dict([(table, "keep_all") for table in full_linking.keys()])
        column_description = schema_manager.get_filtered_schema(db_id, full_linking, schema_format)
        table_description = schema_manager.get_filtered_schema(db_id, table_linking, schema_format)
    except Exception as e:
        logger.warning(f"Error parsing schema linking prediction, returning all: {str(e)}")
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


def run_candidate_schema_linking(
    sample: dict,
    candidate_configs: list[dict],
    schema_manager: SchemaManager,
    azure_linker_token_counter: TokenCounter,
    gcp_linker_token_counter: TokenCounter,
) -> defaultdict:
    """run schema linking for all candidate configs

    Args:
        sample: one sample from the test set json
        candidate_configs: list of candidate configs
        schema_manager: the schema manager
    Returns:
        schema_linking_outputs: defaultdict of schema linking outputs (model > format > outputs)
    """
    jobs: list[dict] = []
    job_configs = []
    schema_linking_outputs = defaultdict(lambda: defaultdict(dict))
    for candidate_config in candidate_configs:
        schema_format = candidate_config["schema_format"]
        schema_linking_generator_name = candidate_config["generator"]
        schema_linking_model = candidate_config["model"]
        config = (schema_linking_model, schema_format)
        # only do unique model & format combinations, as it does both table & column
        if config not in job_configs:
            if schema_linking_generator_name == "azure":
                schema_linking_generator = AzureGenerator(
                    model=schema_linking_model,
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    counter=azure_linker_token_counter,
                )
            elif schema_linking_generator_name == "gcp":
                schema_linking_generator = GCPGenerator(
                    model=schema_linking_model,
                    api_key=os.getenv("GCP_KEY"),
                    counter=gcp_linker_token_counter,
                )
            else:
                raise ValueError(f"Invalid generator: {schema_linking_generator_name}")
            job_configs.append(config)
            jobs.append(
                {
                    "generator": schema_linking_generator,
                    "schema_manager": schema_manager,
                    "sample": sample,
                    "schema_format": schema_format,
                    "schema_linking_generator": schema_linking_generator,
                }
            )

    def run_schema_linking_wrapper(job_dict):
        return run_schema_linking(**job_dict)

    with ThreadPool(len(jobs)) as pool:
        schema_linking_predictions = list(pool.imap(run_schema_linking_wrapper, jobs))

    for idx in range(len(job_configs)):
        model, fmt = job_configs[idx]
        schema_linking_outputs[model][fmt] = schema_linking_predictions[idx]

    return schema_linking_outputs


def run_fewshot_retrieval(
    embedding: list[float],
    retriever: LocalRetriever,
    top_k: int = 3,
) -> list[dict]:
    """run the fewshot retrieval task

    Args:
        embedding: the embedding of the question
        retriever: retriever pre-loaded with vectors and data
        top_k: number of top k results to return
    Returns:
        results: list of dict with the top k results (with keys id, distance, data)
    """
    return retriever.query(embedding, top_k=top_k)


def run_sql_generation(
    generator: BaseGenerator,
    sample: dict,
    few_shot_results: list[dict],
    schema_linking_outputs: dict,
    schema_format: str,
    schema_filtering: Literal["none", "table", "column"],
) -> str:
    """run the sql generation task"""
    # format messages
    few_shot_examples: list[dict] = [result for result in few_shot_results if schema_format in result["data"]]
    message_formatter = GenaCoTwEvidencePromptFormatter(
        database_type="sqlite",
        few_shot_query_key="question",
        few_shot_target_key="SQL",
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
    return sql_prediction, messages


def run_candidate_sql_generation(
    sample: dict,
    generator: BaseGenerator,
    candidate_configs: list[dict],
    schema_linking_outputs: defaultdict,
    few_shot_results: list[dict],
) -> list[str]:

    # built args
    gen_args: list[dict] = []
    for candidate_config in candidate_configs:
        model = candidate_config["model"]
        schema_format = candidate_config["schema_format"]
        schema_filtering = candidate_config["schema_filtering"]
        schema_linking_output = schema_linking_outputs[model][schema_format]

        gen_args.append(
            {
                "generator": generator,
                "sample": sample,
                "few_shot_results": few_shot_results,
                "schema_linking_outputs": schema_linking_output,
                "schema_format": schema_format,
                "schema_filtering": schema_filtering,
            }
        )

    def run_sql_generation_wrapper(job_dict) -> str:
        try:
            return run_sql_generation(**job_dict)
        except Exception as e:
            # handle e.g. the StopCandidateException due to RECITATION
            logger.warning(f"Error generating SQL: {type(e).__name__}: {str(e)}")
            return "", []

    with ThreadPool(len(gen_args)) as pool:
        results: list[tuple[str, list[str]]] = list(pool.imap(run_sql_generation_wrapper, gen_args))
        # get candidate sqls and messages
        candidate_sqls = [sql for sql, _ in results]
        messages = [message for _, message in results]

    return candidate_sqls, messages


def run_sql_rewrite(
    generator: BaseGenerator,
    question: str,
    original_sql: str,
    schema_description: str,
) -> tuple[str, list[dict]]:
    """run the sql generation task"""
    # format messages
    message_formatter = RewritePromptFormatter(
        database_type="sqlite",
    )
    messages = message_formatter.generate_messages(
        schema_description=schema_description,
        query=question,
        predicted_sql=original_sql,
    )
    # run inference
    raw_prediction = generator.generate(messages, temperature=0.0)
    sql_prediction = extract_first_code_block(raw_prediction)
    if not sql_prediction:
        sql_prediction = raw_prediction
    return sql_prediction, messages


def check_need_rewrite(result):
    if len(result) == 0:
        return True
    else:
        has_non_none = False
        for result_row in result:
            for value in result_row.values():
                if value is not None and value != "" and value != [] and value != 0 and value != 0.0:
                    has_non_none = True
                    break
            if has_non_none:
                break
        if not has_non_none:
            return True
    return False


def get_filtered_schema_description_for_rewrite(db_name, schema, prediction):
        table_names = get_table_names_from_query(prediction)
        filtered_schema = {"tables": {}}
        for table_name in table_names:
            table_name = table_name.lower()
            if table_name in schema["tables"]:
                filtered_schema["tables"][table_name] = schema["tables"][table_name]
        return schema_to_datagrip_format(db_name, filtered_schema)


def run_rewrite_check(sample: dict, predicted_sql: str, dataset: BaseDataset, generator: BaseGenerator, schema_manager: SchemaManager) -> tuple[str, list[dict]]:
    database = sample["db_id"]
    max_retries = 3
    attempt = 0
    current_sql = predicted_sql
    all_messages = []
    
    while attempt < max_retries:
        # Check current SQL execution
        execution_result_dict: dict = dataset.validate_query(database, current_sql)
        execution_results: list[dict] = execution_result_dict.get("execution_result", [])
        
        # If no rewrite needed, return current SQL
        if not check_need_rewrite(execution_results):
            return current_sql, all_messages
            
        # Get filtered schema for rewrite
        filtered_schema_description = get_filtered_schema_description_for_rewrite(
            database,
            schema_manager.get_schema_mapping(database), 
            current_sql
        )
        
        try:
            # Attempt rewrite
            rewritten_sql, messages = run_sql_rewrite(
                generator, 
                sample["question"], 
                current_sql, 
                filtered_schema_description
            )
            all_messages.extend(messages)
            
            current_sql = rewritten_sql
            logger.debug(f"Rewrite attempt {attempt + 1}, SQL: {current_sql}")
            
        except Exception as e:
            logger.error(f"Error in run_sql_rewrite attempt {attempt + 1}: {str(e)}")
            break
            
        attempt += 1

    return current_sql, all_messages


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
        execution_result_dict: dict = dataset.validate_query(database, sql_query)
        execution_results: list[dict] = execution_result_dict.get("execution_result", [])
        is_valid = execution_result_dict.get("validated", False)

        sample_dicts.append(
            {
                "sql": sql_query,
                "valid": is_valid,
                "results": execution_results,
            }
        )
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


def save_token_counts(output_dir: str, counters: dict) -> None:
    """Save token counts to a JSON file in the specified directory.
    
    Args:
        output_dir: Directory to save the token counts
        counters: Dictionary of counters to save
    """
    os.makedirs(output_dir, exist_ok=True)
    counts_file = os.path.join(output_dir, "token_counts.json")
    counts = {name: counter.get_counts() for name, counter in counters.items()}
    with open(counts_file, "w") as f:
        json.dump(counts, f, indent=2)
    logger.info(f"Saved token counts to {counts_file}")


def load_token_counts(output_dir: str, counters: dict) -> None:
    """Load token counts from a JSON file in the specified directory.
    
    Args:
        output_dir: Directory containing the token counts
        counters: Dictionary of counters to update
    """
    counts_file = os.path.join(output_dir, "token_counts.json")
    if os.path.exists(counts_file):
        with open(counts_file, "r") as f:
            counts = json.load(f)
        for name, counter in counters.items():
            if name in counts:
                counter.update_counts(counts[name])
        logger.info(f"Loaded token counts from {counts_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-database-path",
        type=str,
        required=True,
        help="path to the test databases base directory",
    )
    parser.add_argument(
        "--test-json-path",
        type=str,
        required=True,
        help="path to the test.json file",
    )
    parser.add_argument(
        "--test-tables-json-path",
        type=str,
        required=True,
        help="path to the test_tables.json file",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default="./bird_data/valid_multi_table_queries_080425_embeddings.npy",
        help="path to preprocessed numpy embeddings file",
    )
    parser.add_argument(
        "--embeddings-data-path",
        type=str,
        default="./bird_data/valid_multi_table_queries_080425.json",
        help="path to preprocessed json embeddings data file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        default="../outputs",
        help="target output path",
    )
    parser.add_argument(
        "--candidate-configs-path",
        type=str,
        default="./bird_data/consistency_candidate_configs.yaml",
        help="path to the candidate configs file",
    )
    parser.add_argument(
        "--column-meaning-json-path",
        type=str,
        default=None,
        help="path to the column_meaning.json file, leave blank if not used",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="run in debug mode (do small subset of data, default is None)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of workers to use for inference, default is 3",
    )
    # make it a boolean
    parser.add_argument(
        "--save-messages",
        action="store_true",
        default=False,
        help="save messages to separate files for debugging",
    )
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

    # if output path does not exist, create it
    if not os.path.isdir(args.output_path):
        logger.info(f"Output directory not found, creating it: {args.output_path}")
        os.makedirs(args.output_path)
    else:
        logger.info(f"Output directory found, existing outputs will be overwritten: {args.output_path}")

    # load candidate configs
    top_k = 3  # 3 by default, can override in candidate configs
    with open(args.candidate_configs_path, "r") as f:
        candidate_config_data: list[dict] = yaml.safe_load(f)
        if "configs" not in candidate_config_data:
            raise ValueError("candidate_config_data must contain a 'configs' key")
        if "top_k" in candidate_config_data:
            top_k = candidate_config_data["top_k"]
        candidate_configs: list[dict] = candidate_config_data["configs"]

    # verify candidate config keys:
    for config in candidate_configs:
        assert "schema_format" in config
        assert "schema_filtering" in config
        assert "generator" in config
        assert "model" in config

    # load test.json
    logger.info("Loading test data...")
    with open(args.test_json_path, "r") as f:
        test_data: list[dict] = json.load(f)

    # load column_meaning.json
    if args.column_meaning_json_path is not None:
        with open(args.column_meaning_json_path, "r") as f:
            column_meaning_json = json.load(f)
    else:
        column_meaning_json = {}

    # counters for tracking usage
    cohere_character_counter = CharacterCounter()
    azure_linker_token_counter = TokenCounter()
    gcp_linker_token_counter = TokenCounter()
    gcp_candidate_token_counter = TokenCounter() 
    gcp_selection_token_counter = TokenCounter()

    # create generators
    logger.info("Creating embedder...")
    embedder = BedrockCohereEmbedder(
        model=os.getenv("AWS_MODEL_NAME"),
        region_name=os.getenv("AWS_REGION_NAME"),
        input_type=os.getenv("AWS_INPUT_TYPE"),
        counter=cohere_character_counter,
        sleep_ms=10,
    )

    logger.info("Creating & testing azure generator...")
    test_messages = [{"role": "user", "content": "What is the capital of South Korea? Answer in one word."}]

    test_azure_generator = AzureGenerator(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    p = test_azure_generator.generate(test_messages, temperature=0.0)
    logger.info(f"Azure generator test response: '{p}'")

    logger.info("Creating & testing Gemini generator...")
    gcp_generator_candidate = GCPGenerator(
        model="gemini-1.5-flash",
        api_key=os.getenv("GCP_KEY"),
        counter=gcp_candidate_token_counter,
    )
    gcp_generator_selection = GCPGenerator(
        model="gemini-1.5-flash",
        api_key=os.getenv("GCP_KEY"),
        counter=gcp_selection_token_counter,
    )

    p = gcp_generator_candidate.generate(test_messages, temperature=0.0)
    logger.info(f"Gemini generator test response: '{p}'")

    #############################
    # preprocessing
    #############################
    dataset, schema_manager = prepare_dataset_information(args.test_database_path, args.test_tables_json_path)
    retriever = prepare_fewshot_retriever(args.embeddings_path, args.embeddings_data_path)

    logger.info("Preprocessing complete, starting inference...")

    # check for debug mode
    if args.debug:
        logger.warning(f"!!!!! DEBUG - Running on {args.debug} data subset !!!!!")
        test_data = test_data[: args.debug]
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    #############################
    # schema linking
    #############################
    # load any existing schema linking jsons
    schema_linking_output_dir = os.path.join(args.output_path, "1_schema_linking")
    os.makedirs(schema_linking_output_dir, exist_ok=True)
    
    # Load token counts for schema linking
    schema_linking_counters = {
        "azure_linker": azure_linker_token_counter,
        "gcp_linker": gcp_linker_token_counter
    }
    load_token_counts(schema_linking_output_dir, schema_linking_counters)
    
    cached_schema_linking_results: dict = {}
    for file in os.listdir(schema_linking_output_dir):
        if file.endswith(".json") and file != "token_counts.json":
            # get id from filename
            index = int(file.split(".")[0])
            with open(os.path.join(schema_linking_output_dir, file), "r") as f:
                cached_schema_linking_results[index] = json.load(f)
    logger.info(f"Loaded {len(cached_schema_linking_results)} cached schema linking results")
    logger.info(f"Running schema linking for {len(test_data)-len(cached_schema_linking_results)} samples")
    
    # run schema linking for each sample, with threading executor
    schema_linking_outputs: dict = {}
    def maybe_run_candidate_schema_linking(
            idx, 
            cached_schema_linking_results, 
            sample, 
            candidate_configs, 
            schema_manager, 
            azure_linker_token_counter, 
            gcp_linker_token_counter
        ) -> dict:
        if idx in cached_schema_linking_results:
            schema_linking_result = cached_schema_linking_results[idx]
            # DENI HACK
            # for model in schema_linking_result.keys():
            #     for mode in schema_linking_result[model].keys():
            #         vals = schema_linking_result[model][mode]
            #         column_description = schema_manager.get_filtered_schema(sample["db_id"], vals["column_linking"], mode)
            #         table_description = schema_manager.get_filtered_schema(sample["db_id"], vals["table_linking"], mode)
            #         schema_linking_result[model][mode]["column_description"] = column_description
            #         schema_linking_result[model][mode]["table_description"] = table_description
            # with open(os.path.join(schema_linking_output_dir, f"{idx:4d}.json"), "w") as f:
            #     json.dump(schema_linking_result, f, indent=2)
        else:
            schema_linking_result = run_candidate_schema_linking(
                sample, 
                candidate_configs, 
                schema_manager, 
                azure_linker_token_counter, 
                gcp_linker_token_counter,
            )
            with open(os.path.join(schema_linking_output_dir, f"{idx:4d}.json"), "w") as f:
                json.dump(schema_linking_result, f, indent=2)
        return dict(schema_linking_result)
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(
            maybe_run_candidate_schema_linking, 
            idx, 
            cached_schema_linking_results, 
            sample, 
            candidate_configs, 
            schema_manager, 
            azure_linker_token_counter, 
            gcp_linker_token_counter
        ) for idx, sample in enumerate(test_data)]
        for idx, future in enumerate(tqdm.tqdm(futures, total=len(test_data))):
            schema_linking_outputs[idx] = future.result()
    
    # Save token counts for schema linking
    save_token_counts(schema_linking_output_dir, schema_linking_counters)

    del cached_schema_linking_results
    logger.info("Schema linking complete")

    #############################
    # question embedding
    #############################
    embedding_output_dir = os.path.join(args.output_path, "2_embeddings")
    os.makedirs(embedding_output_dir, exist_ok=True)
    
    # Load token counts for embeddings
    embedding_counters = {
        "cohere_character": cohere_character_counter
    }
    load_token_counts(embedding_output_dir, embedding_counters)
    
    # check for cached npy embeddings file
    _embeddings = None
    if os.path.isfile(os.path.join(embedding_output_dir, "embeddings.npy")):
        _embeddings = np.load(os.path.join(embedding_output_dir, "embeddings.npy"))
    if _embeddings is not None and len(_embeddings) >= len(test_data):
        embeddings = _embeddings.tolist()
        logger.info(f"Loaded cached embeddings of size ({np.array(embeddings).shape})")
    else:
        logger.info("Generating embeddings:")
        # try to load processed questions
        if os.path.isfile(os.path.join(embedding_output_dir, "masked_questions.txt")):
            with open(os.path.join(embedding_output_dir, "masked_questions.txt"), "r") as f:
                masked_questions = f.read().splitlines()
        else:
            logger.info("Processing questions...")
            test_questions = [sample["question"] for sample in test_data]
            masked_questions = [replace_entities_with_tokens(question) for question in tqdm.tqdm(test_questions)]
            print(f"{len(test_questions)=}")
            print(f"{len(masked_questions)=}")
            with open(os.path.join(embedding_output_dir, "masked_questions.txt"), "w") as f:
                f.write("\n".join(masked_questions))
        logger.info("Embedding processed questions...")
        embeddings = embedder.embed(masked_questions, verbose=True)
        np.save(os.path.join(embedding_output_dir, "embeddings.npy"), embeddings)
    
    # Save token counts for embeddings
    save_token_counts(embedding_output_dir, embedding_counters)
    
    logger.info(f"Embeddings of size ({np.array(embeddings).shape}) complete")
    
    #############################
    # few-shot retrieval
    #############################
    fewshot_retrieval_output_dir = os.path.join(args.output_path, "3_fewshot_retrieval")
    os.makedirs(fewshot_retrieval_output_dir, exist_ok=True)
    
    cached_fewshot_retrieval_results: dict = {}
    for file in os.listdir(fewshot_retrieval_output_dir):
        if file.endswith(".json") and file != "token_counts.json":
            index = int(file.split(".")[0])
            with open(os.path.join(fewshot_retrieval_output_dir, file), "r") as f:
                cached_fewshot_retrieval_results[index] = json.load(f)
    logger.info(f"Loaded {len(cached_fewshot_retrieval_results)} cached fewshot retrieval results")
    logger.info(f"Running fewshot retrieval for {len(test_data)-len(cached_fewshot_retrieval_results)} samples")
    # run fewshot retrieval for each sample
    fewshot_retrieval_results: dict = {}
    for idx, sample in enumerate(tqdm.tqdm(test_data)):
        if idx in cached_fewshot_retrieval_results:
            fewshot_retrieval_result: list[dict] = cached_fewshot_retrieval_results[idx]
        else:
            fewshot_retrieval_result: list[dict] = run_fewshot_retrieval(embeddings[idx], retriever, top_k=top_k)
        fewshot_retrieval_results[idx] = fewshot_retrieval_result
        with open(os.path.join(fewshot_retrieval_output_dir, f"{idx:4d}.json"), "w") as f:
            json.dump(fewshot_retrieval_result, f, indent=2)
    del cached_fewshot_retrieval_results
    logger.info("Fewshot retrieval complete")

    #############################
    # sql candidate generation
    #############################
    sql_generation_output_dir = os.path.join(args.output_path, "4_sql_generation")
    os.makedirs(sql_generation_output_dir, exist_ok=True)
    
    # Load token counts for SQL generation
    sql_generation_counters = {
        "gcp_candidate": gcp_candidate_token_counter
    }
    load_token_counts(sql_generation_output_dir, sql_generation_counters)
    
    cached_sql_generation_results: dict = {}
    for file in os.listdir(sql_generation_output_dir):
        if file.endswith(".json") and file != "token_counts.json" and not file.endswith("_messages.json"):
            index = int(file.split(".")[0])
            with open(os.path.join(sql_generation_output_dir, file), "r") as f:
                cached_sql_generation_results[index] = json.load(f)
    logger.info(f"Loaded {len(cached_sql_generation_results)} cached sql generation results")
    logger.info(f"Running sql generation for {len(test_data)-len(cached_sql_generation_results)} samples")
    # run sql generation for each sample, using threading executor
    def maybe_run_candidate_sql_generation(
            idx, 
            cached_sql_generation_results, 
            sample, 
            generator, 
            candidate_configs, 
            schema_linking_outputs, 
            few_shot_results
        ) -> list[str]:
        if idx in cached_sql_generation_results:
            sql_generation_result: list[str] = cached_sql_generation_results[idx]
            messages = []
        else:
            sql_generation_result, messages = run_candidate_sql_generation(
                sample=sample,
                generator=generator,
                candidate_configs=candidate_configs,
                schema_linking_outputs=schema_linking_outputs,
                few_shot_results=few_shot_results,
            )
        with open(os.path.join(sql_generation_output_dir, f"{idx:4d}.json"), "w") as f:
            json.dump(sql_generation_result, f, indent=2)
        if messages and args.save_messages:
            with open(os.path.join(sql_generation_output_dir, f"{idx:4d}_messages.json"), "w") as f:
                json.dump(messages, f, indent=2)
        return sql_generation_result

    sql_generation_results: dict = {}
    with ThreadPoolExecutor(max_workers=min(4, args.num_workers)) as executor:
        futures = [executor.submit(
            maybe_run_candidate_sql_generation, 
            idx, 
            cached_sql_generation_results, 
            sample, 
            gcp_generator_candidate, 
            candidate_configs, 
            schema_linking_outputs[idx], 
            fewshot_retrieval_results[idx]
        ) for idx, sample in enumerate(test_data)]
        for idx, future in enumerate(tqdm.tqdm(futures, total=len(test_data))):
            sql_generation_results[idx] = future.result()
    
    # Save token counts for SQL generation
    save_token_counts(sql_generation_output_dir, sql_generation_counters)

    del cached_sql_generation_results
    logger.info("Sql generation complete")

    #############################
    # rewriting
    #############################
    rewritten_results_output_dir = os.path.join(args.output_path, "5_rewritten_results")
    os.makedirs(rewritten_results_output_dir, exist_ok=True)
    
    # Load token counts for rewrite check
    rewrite_counters = {
        "gcp_candidate": gcp_candidate_token_counter
    }
    load_token_counts(rewritten_results_output_dir, rewrite_counters)
    
    cached_rewritten_results: dict = {}
    for file in os.listdir(rewritten_results_output_dir):
        if file.endswith(".json") and file != "token_counts.json" and not file.endswith("_messages.json"):
            index = int(file.split(".")[0])
            with open(os.path.join(rewritten_results_output_dir, file), "r") as f:
                cached_rewritten_results[index] = json.load(f)
    logger.info(f"Loaded {len(cached_rewritten_results)} cached rewritten results")
    logger.info(f"Running rewrite check for {len(test_data)-len(cached_rewritten_results)} samples")
    
    # Flatten the sql_generation_results dict into a list of tuples (idx, sql_idx, sql)
    flattened_sqls = []
    for idx, sqls in sql_generation_results.items():
        for sql_idx, sql in enumerate(sqls):
            flattened_sqls.append((idx, sql_idx, sql))
    
    # Function to run rewrite check on a single SQL
    def run_rewrite_check_wrapper(params):
        idx, sql_idx, sql = params
        sample = test_data[idx]
        try:
            rewritten_sql, messages = run_rewrite_check(
                sample=sample,
                predicted_sql=sql,
                dataset=dataset,
                generator=gcp_generator_candidate,
                schema_manager=schema_manager
            )
            return idx, sql_idx, rewritten_sql, messages
        except Exception as e:
            logger.error(f"Error in rewrite check for idx {idx}, sql_idx {sql_idx}: {str(e)}")
            return idx, sql_idx, sql, []  # Return original SQL if rewrite fails
    
    # Run rewrite check in parallel
    rewritten_sqls = {}
    rewrite_messages = {}
    if flattened_sqls:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(run_rewrite_check_wrapper, params) for params in flattened_sqls]
            for future in tqdm.tqdm(futures, total=len(flattened_sqls), desc="Running rewrite check"):
                idx, sql_idx, rewritten_sql, messages = future.result()
                if idx not in rewritten_sqls:
                    rewritten_sqls[idx] = [""] * len(sql_generation_results[idx])
                    rewrite_messages[idx] = [[] for _ in range(len(sql_generation_results[idx]))]
                rewritten_sqls[idx][sql_idx] = rewritten_sql
                rewrite_messages[idx][sql_idx] = messages
    
    # Save rewritten results
    for idx, rewritten_sql_list in rewritten_sqls.items():
        with open(os.path.join(rewritten_results_output_dir, f"{idx:04d}.json"), "w") as f:
            json.dump(rewritten_sql_list, f, indent=2)
        
        # Save messages if enabled
        if args.save_messages and idx in rewrite_messages:
            with open(os.path.join(rewritten_results_output_dir, f"{idx:04d}_messages.json"), "w") as f:
                json.dump(rewrite_messages[idx], f, indent=2)
    
       # Calculate and log rewrite statistics
    total_sqls = sum(len(sqls) for sqls in sql_generation_results.values())
    rewritten_sqls_count = 0
    
    for idx, original_sqls in sql_generation_results.items():
        rewritten_sqls_list = rewritten_sqls.get(idx, [])
        for i, original_sql in enumerate(original_sqls):
            if i < len(rewritten_sqls_list) and rewritten_sqls_list[i] != original_sql:
                rewritten_sqls_count += 1
    
    rewrite_percentage = (rewritten_sqls_count / total_sqls) * 100 if total_sqls > 0 else 0
    logger.info(f"Rewrite statistics: {rewritten_sqls_count} out of {total_sqls} SQL queries needed rewriting ({rewrite_percentage:.2f}%)")
    
    
    # Save token counts for rewrite check
    save_token_counts(rewritten_results_output_dir, rewrite_counters)
    
    logger.info("Rewrite check complete")

    #############################
    # candidate selection
    #############################
    candidate_selection_output_dir = os.path.join(args.output_path, "6_candidate_selection")
    os.makedirs(candidate_selection_output_dir, exist_ok=True)
    
    # Load token counts for candidate selection
    selection_counters = {
        "gcp_selection": gcp_selection_token_counter
    }
    load_token_counts(candidate_selection_output_dir, selection_counters)
    
    cached_candidate_selection_results: dict = {}
    for file in os.listdir(candidate_selection_output_dir):
        if file.endswith(".txt") and file != "token_counts.json":
            index = int(file.split(".")[0])
            with open(os.path.join(candidate_selection_output_dir, file), "r") as f:
                cached_candidate_selection_results[index] = f.read()
    logger.info(f"Loaded {len(cached_candidate_selection_results)} cached candidate selection results")
    logger.info(f"Running candidate selection for {len(test_data)-len(cached_candidate_selection_results)} samples")
    # run candidate selection for each sample
    candidate_selection_results: dict = {}
    for idx, sample in enumerate(tqdm.tqdm(test_data)):
        if idx in cached_candidate_selection_results:
            candidate_selection_result = cached_candidate_selection_results[idx]
        else:
            # Use rewritten SQLs if available, otherwise use original SQLs
            candidate_sqls = rewritten_sqls.get(idx, sql_generation_results[idx])
            candidate_selection_result = run_candidate_selection(
                dataset=dataset,
                schema_manager=schema_manager,
                sample=sample,
                candidate_sqls=candidate_sqls,
                generator=gcp_generator_selection,
                chase=True,
            )
        candidate_selection_results[idx] = candidate_selection_result
        with open(os.path.join(candidate_selection_output_dir, f"{idx:4d}.txt"), "w") as f:
            f.write(candidate_selection_result)
    
    # Save token counts for candidate selection
    save_token_counts(candidate_selection_output_dir, selection_counters)
    
    logger.info("Candidate selection complete")

    #############################
    # output saving
    #############################
    # check idx == question_id and add \t----- bird -----\t<db_id>
    predictions = {}
    for idx in range(len(candidate_selection_results)):
        question_id = test_data[idx]["question_id"]
        db_id = test_data[idx]["db_id"]
        prediction = candidate_selection_results[idx]
        predictions[str(question_id)] = prediction + f"\t----- bird -----\t{db_id}"
    if len(predictions) != len(test_data):
        raise ValueError(f"predictions length ({len(predictions)}) does not match test data length ({len(test_data)})")
    with open(os.path.join(args.output_path, "predict.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    total_dict = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
    }

    for token_counter in [
        azure_linker_token_counter,
        gcp_linker_token_counter,
        gcp_candidate_token_counter,
        gcp_selection_token_counter,
    ]:
        total_dict["prompt_tokens"] += token_counter.prompt_tokens
        total_dict["completion_tokens"] += token_counter.completion_tokens
        total_dict["total_tokens"] += token_counter.total_tokens
        total_dict["call_count"] += token_counter.call_count

    all_counts = {
        "cohere_character_counts": cohere_character_counter.get_counts(),
        "azure_linker_counts": azure_linker_token_counter.get_counts(),
        "gcp_linker_counts": gcp_linker_token_counter.get_counts(),
        "gcp_candidate_counts": gcp_candidate_token_counter.get_counts(),
        "gcp_selection_counts": gcp_selection_token_counter.get_counts(),
        "total_token_usage": total_dict,
    }
    logger.info(f"Token Counts: {json.dumps(all_counts, indent=4)}")
    with open(os.path.join(args.output_path, "token_counts.json"), "w") as f:
        json.dump(all_counts, f, indent=2)


if __name__ == "__main__":
    main()
