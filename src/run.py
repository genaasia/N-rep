import argparse
import json
import os
import sys
import warnings

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from typing import Literal

import dill as pickle
import numpy as np
import tqdm
import yaml

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from bird_data.schema_linking_data import SCHEMA_LINKING_EXAMPLES

from text2sql.data import BaseDataset, SqliteDataset, SchemaManager
from text2sql.data.schema_to_text import schema_to_datagrip_format
from text2sql.engine.embeddings import BaseEmbedder, BedrockCohereEmbedder, EmbeddingResult
from text2sql.engine.generation import BaseGenerator, AzureGenerator, GCPGenerator, GenerationResult
from text2sql.engine.prompts.formatters import GenaCoTwEvidencePromptFormatter
from text2sql.engine.prompts.formatters import SchemaLinkingFewShotFormatter
from text2sql.engine.prompts.formatters import RewritePromptFormatter
from text2sql.engine.retrieval import LocalRetriever
from text2sql.engine.generation.postprocessing import extract_first_code_block
from text2sql.utils.postprocess import get_table_names_from_query
from text2sql.utils import parse_json_from_prediction, replace_entities_with_tokens

from text2sql.pipeline.selection import select_best_candidate


class SchemaLinkingOutput(BaseModel):
    question_id: int
    model_name: str
    schema_format: str
    messages: list[dict]
    generator_output: GenerationResult
    prediction: str
    table_linking: dict | None
    column_linking: dict | None
    table_description: str
    column_description: str
    full_description: str


class CandidateGenerationOutput(BaseModel):
    question_id: int
    schema_format: str
    schema_filtering: Literal["none", "table", "column"]
    messages: list[dict]
    generator_output: GenerationResult
    candidate_sql: str


class CandidateGenerationOutputBundle(BaseModel):
    question_id: int
    candidate_configs: list[dict]
    candidates: list[CandidateGenerationOutput]


class RewriteOutput(BaseModel):
    question_id: int
    original_sql: str
    rewritten_sql: str
    is_rewritten: bool
    messages: list[dict]
    all_messages: list[dict] = []
    generator_output: GenerationResult | None


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


def run_embedding(
    embedder: BaseEmbedder,
    samples: list[dict],
    output_dir: str,
    n_jobs: int = 1,
) -> list[EmbeddingResult]:
    """run the embedding task

    Args:
        embedder: the embedder
        samples: the samples to embed
        output_dir: the output directory
        n_jobs: the number of jobs to run in parallel
    Returns:
        embedding_result: the embedding result
    """
    # run embedding in parallel using thread executor,
    # and save each EmbeddingResult to a file named "question_id-{question_id:04d}.json" using model_dump_json
    # use tqdm to show progress
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(embedder.embed, [sample["question"]], verbose=False) for sample in samples]
        question_ids = [sample["question_id"] for sample in samples]
        responses: list[EmbeddingResult] = []
        for idx, future in tqdm.tqdm(enumerate(futures), total=len(samples)):
            embedding_result = future.result()
            question_id = question_ids[idx]
            with open(os.path.join(output_dir, f"embedding_qid-{question_id:04d}.json"), "w") as f:
                f.write(embedding_result.model_dump_json(indent=2))
            responses.append(embedding_result)
    return responses


def run_schema_linking(
    generator: BaseGenerator,
    schema_manager: SchemaManager,
    model_name: str,
    sample: dict,
    schema_format: str,
    schema_linking_generator: str,
) -> SchemaLinkingOutput:
    """run the schema linking task

    Args:
        generator: LLM generator
        schema_manager: the schema manager
        model_name: the name of the model
        sample: one sample from the test set json
        schema_format: schema mode to use
    Returns:
        outputs: dict with the table_linking, column_linking, table_description and column_description
    """
    db_id = sample["db_id"]
    question = sample["question"]
    evidence = sample.get("evidence", "")
    question_id = sample["question_id"]

    # generate the input messages and run inference
    message_formatter = SchemaLinkingFewShotFormatter(SCHEMA_LINKING_EXAMPLES, description_format=schema_format)
    schema_description = schema_manager.get_full_schema(db_id, schema_format)
    is_gemini = schema_linking_generator == "gcp"
    messages: list[dict] = message_formatter.generate_messages(schema_description, question, evidence, gemini=is_gemini)
    prediction_output: GenerationResult = generator.generate(messages, temperature=0.0)
    raw_prediction: str = prediction_output.text
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

    return SchemaLinkingOutput(
        question_id=question_id,
        model_name=model_name,
        schema_format=schema_format,
        messages=messages,
        generator_output=prediction_output,
        prediction=raw_prediction,
        table_linking=table_linking,
        column_linking=full_linking,
        table_description=table_description,
        column_description=column_description,
        full_description=schema_description,
    )


def run_candidate_schema_linking(
    sample: dict,
    candidate_configs: list[dict],
    schema_manager: SchemaManager,
) -> list[SchemaLinkingOutput]:
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
                )
            elif schema_linking_generator_name == "gcp":
                schema_linking_generator = GCPGenerator(
                    model=schema_linking_model,
                    api_key=os.getenv("GCP_KEY"),
                )
            else:
                raise ValueError(f"Invalid generator: {schema_linking_generator_name}")
            job_configs.append(config)
            jobs.append(
                {
                    "generator": schema_linking_generator,
                    "model_name": schema_linking_model,
                    "schema_manager": schema_manager,
                    "sample": sample,
                    "schema_format": schema_format,
                    "schema_linking_generator": schema_linking_generator,
                }
            )

    def run_schema_linking_wrapper(job_dict):
        return run_schema_linking(**job_dict)

    with ThreadPool(len(jobs)) as pool:
        schema_linking_predictions: list[SchemaLinkingOutput] = list(pool.imap(run_schema_linking_wrapper, jobs))

    return schema_linking_predictions


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
    schema_linking_outputs: SchemaLinkingOutput,
    schema_format: str,
    schema_filtering: Literal["none", "table", "column"],
) -> CandidateGenerationOutput:
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
        schema_description = schema_linking_outputs.table_description
    elif schema_filtering == "column":
        schema_description = schema_linking_outputs.column_description
    else:
        schema_description = schema_linking_outputs.full_description
    messages = message_formatter.generate_messages(
        schema_description=schema_description,
        query=sample["question"],
        evidence=sample.get("evidence", ""),
        few_shot_examples=few_shot_examples,
    )
    # run inference
    prediction_output: GenerationResult = generator.generate(messages, temperature=0.0)
    raw_prediction = prediction_output.text
    sql_prediction = extract_first_code_block(raw_prediction)
    if not sql_prediction:
        sql_prediction = raw_prediction
    return CandidateGenerationOutput(
        question_id=sample["question_id"],
        schema_format=schema_format,
        schema_filtering=schema_filtering,
        messages=messages,
        generator_output=prediction_output,
        candidate_sql=sql_prediction,
    )


def run_candidate_sql_generation(
    sample: dict,
    generator: BaseGenerator,
    candidate_configs: list[dict],
    candidate_schema_linking_outputs: dict[str, dict[str, SchemaLinkingOutput]],
    few_shot_results: list[dict],
) -> list[CandidateGenerationOutput]:
    """run the candidate sql generation task

    Args:
        sample: the sample to run the task on
        generator: the generator to use
        candidate_configs: the candidate configs to use
        candidate_schema_linking_outputs: the candidate schema linking outputs for this question id
        few_shot_results: the few shot results to use
    Returns:
        candidate_sqls: the candidate sqls
    """
    # built args
    gen_args: list[dict] = []
    for candidate_config in candidate_configs:
        model = candidate_config["model"]
        schema_format = candidate_config["schema_format"]
        schema_filtering = candidate_config["schema_filtering"]
        schema_linking_output: SchemaLinkingOutput = candidate_schema_linking_outputs[model][schema_format]

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
        results: list[CandidateGenerationOutput] = list(pool.imap(run_sql_generation_wrapper, gen_args))

    return results


def run_sql_rewrite(
    generator: BaseGenerator,
    sample: dict,
    original_sql: str,
    schema_description: str,
) -> RewriteOutput:
    """run the sql generation task"""
    # format messages
    message_formatter = RewritePromptFormatter(
        database_type="sqlite",
    )
    messages = message_formatter.generate_messages(
        schema_description=schema_description,
        query=sample["question"],
        predicted_sql=original_sql,
    )
    # run inference
    prediction_output: GenerationResult = generator.generate(messages, temperature=0.0)
    raw_prediction = prediction_output.text
    sql_prediction = extract_first_code_block(raw_prediction)
    if not sql_prediction:
        sql_prediction = raw_prediction
    return RewriteOutput(
        question_id=sample["question_id"],
        original_sql=original_sql,
        rewritten_sql=sql_prediction,
        messages=messages,
        generator_output=prediction_output,
        is_rewritten=True if sql_prediction != original_sql else False,
    )


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


def run_rewrite_check(
    sample: dict, predicted_sql: str, dataset: BaseDataset, generator: BaseGenerator, schema_manager: SchemaManager
) -> RewriteOutput:
    database = sample["db_id"]
    max_retries = 3
    attempt = 0
    current_sql = predicted_sql
    all_messages = []

    # default output
    rewritten_output = RewriteOutput(
        question_id=sample["question_id"],
        original_sql=predicted_sql,
        rewritten_sql=predicted_sql,
        messages=[],
        generator_output=None,
        is_rewritten=False,
    )

    while attempt < max_retries:
        # Check current SQL execution
        execution_result_dict: dict = dataset.validate_query(database, current_sql)
        execution_results: list[dict] = execution_result_dict.get("execution_result", [])

        # If no rewrite needed, return current SQL
        if not check_need_rewrite(execution_results):
            return rewritten_output

        # Get filtered schema for rewrite
        filtered_schema_description = get_filtered_schema_description_for_rewrite(
            database, schema_manager.get_schema_mapping(database), current_sql
        )

        try:
            # Attempt rewrite
            rewritten_output: RewriteOutput = run_sql_rewrite(
                generator, sample, current_sql, filtered_schema_description
            )
            all_messages.extend(rewritten_output.messages)

            current_sql = rewritten_output.rewritten_sql
            logger.debug(f"Rewrite attempt {attempt + 1}, SQL: {current_sql}")

        except Exception as e:
            logger.error(f"Error in run_sql_rewrite attempt {attempt + 1}: {str(e)}")
            break

        attempt += 1

    rewritten_output.all_messages = all_messages

    return rewritten_output


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
        default="./bird_data/valid_multi_table_queries_embeddings.npy",
        help="path to preprocessed numpy embeddings file",
    )
    parser.add_argument(
        "--embeddings-data-path",
        type=str,
        default="./bird_data/valid_multi_table_queries.json",
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
        default=4,
        help="number of workers to use for inference, default is 4",
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

    # create generators
    logger.info("Creating embedder...")
    embedder = BedrockCohereEmbedder(
        model=os.getenv("AWS_MODEL_NAME"),
        region_name=os.getenv("AWS_REGION_NAME"),
        input_type=os.getenv("AWS_INPUT_TYPE"),
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
    )
    gcp_generator_selection = GCPGenerator(
        model="gemini-1.5-flash",
        api_key=os.getenv("GCP_KEY"),
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
    test_question_ids = [sample["question_id"] for sample in test_data]
    logger.debug(f"Test question ids: {test_question_ids}")

    #############################
    # schema linking
    # outputs: list[SchemaLinkingOutput]
    #############################
    # load any existing schema linking jsons
    schema_linking_output_dir = os.path.join(args.output_path, "1_schema_linking")
    os.makedirs(schema_linking_output_dir, exist_ok=True)

    schema_linking_results: dict = {}
    for file in sorted(os.listdir(schema_linking_output_dir)):
        if file.startswith("schema-linking_") and file.endswith(".json"):
            # get question id from filename
            question_id = int(file.rsplit(".", 1)[0].rsplit("-", 1)[-1])
            if question_id in test_question_ids:
                with open(os.path.join(schema_linking_output_dir, file), "r") as f:
                    schema_linking_output: SchemaLinkingOutput = SchemaLinkingOutput.model_validate_json(f.read())
                    model_name = schema_linking_output.model_name
                    schema_format = schema_linking_output.schema_format
                    if question_id not in schema_linking_results:
                        schema_linking_results[question_id] = {}
                    if model_name not in schema_linking_results[question_id]:
                        schema_linking_results[question_id][model_name] = {}
                    schema_linking_results[question_id][model_name][schema_format] = schema_linking_output
                    logger.debug(
                        f"{question_id} model '{model_name}' schema '{schema_format}' from {os.path.basename(file)}"
                    )
                    logger.debug(f"check model keys for {question_id}: {schema_linking_results[question_id].keys()}")
    logger.info(f"Loaded {len(schema_linking_results)} cached schema linking results")
    # check how many samples not in cache based on question_id
    missing_question_ids = set(
        [sample["question_id"] for sample in test_data if sample["question_id"] not in schema_linking_results]
    )
    logger.info(f"Running schema linking for {len(missing_question_ids)} samples")

    # run schema linking for each sample, with threading executor
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                run_candidate_schema_linking,
                sample,
                candidate_configs,
                schema_manager,
            )
            for idx, sample in enumerate(test_data)
            if sample["question_id"] in missing_question_ids
        ]
        for idx, future in enumerate(tqdm.tqdm(futures, total=len(test_data))):
            question_id = test_data[idx]["question_id"]
            predicted_schema_linking_outputs: list[SchemaLinkingOutput] = future.result()
            for output in predicted_schema_linking_outputs:
                model_name = output.model_name
                f_model_name = model_name.replace("_", "").replace(" ", "")
                schema_format = output.schema_format
                f_schema_format = schema_format.replace("_", "").replace(" ", "")
                output_file = f"schema-linking_mdl-{f_model_name}_fmt-{f_schema_format}_qid-{question_id:04d}.json"
                with open(os.path.join(schema_linking_output_dir, output_file), "w") as f:
                    f.write(output.model_dump_json(indent=2))
                if question_id not in schema_linking_results:
                    schema_linking_results[question_id] = {}
                if model_name not in schema_linking_results[question_id]:
                    schema_linking_results[question_id][model_name] = {}
                schema_linking_results[question_id][model_name][schema_format] = output

    # check all question ids are in schema_linking_outputs
    schema_keys = sorted(schema_linking_results.keys())
    logger.debug(f"Schema linking results keys: {schema_keys}")
    for key in schema_keys:
        logger.debug(f"Schema linking results for question {key}: {schema_linking_results[key].keys()}")
    for key in schema_keys:
        mdl_keys = sorted(schema_linking_results[key].keys())
        for mdl_key in mdl_keys:
            logger.debug(
                f"Schema linking results for question {key} model {mdl_key}: {schema_linking_results[key][mdl_key].keys()}"
            )

    for sample in test_data:
        assert sample["question_id"] in schema_linking_results
        for candidate_config in candidate_configs:
            assert candidate_config["model"] in schema_linking_results[sample["question_id"]]
            assert (
                candidate_config["schema_format"]
                in schema_linking_results[sample["question_id"]][candidate_config["model"]]
            )
    logger.info("Schema linking complete")

    #############################
    # question embedding
    # outputs: EmbeddingResult
    #############################
    embedding_output_dir = os.path.join(args.output_path, "2_embeddings")
    os.makedirs(embedding_output_dir, exist_ok=True)
    embedding_results: dict[int, EmbeddingResult] = {}
    for file in os.listdir(embedding_output_dir):
        if os.path.basename(file).startswith("embedding_qid-") and file.endswith(".json"):
            # get id from filename
            question_id = int(file.split(".")[0].split("-")[-1])
            if question_id in test_question_ids:
                with open(os.path.join(embedding_output_dir, file), "r") as f:
                    embedding_results[question_id] = json.load(f)
    logger.info(f"Loaded {len(embedding_results)} cached embedding results")
    missing_samples = [sample for sample in test_data if sample["question_id"] not in embedding_results]
    logger.info(f"Running embedding for {len(missing_samples)} samples")
    # embed each sample and save via run_embedding()
    new_embedding_results: list[EmbeddingResult] = run_embedding(
        embedder,
        missing_samples,
        embedding_output_dir,
        n_jobs=args.num_workers,
    )
    for idx, sample in enumerate(missing_samples):
        question_id = sample["question_id"]
        embedding_results[question_id] = new_embedding_results[idx]
    # assert all question ids are in embedding_results
    for sample in test_data:
        assert sample["question_id"] in embedding_results
    del new_embedding_results
    logger.info("Embedding complete")

    #############################
    # few-shot retrieval
    #############################
    fewshot_retrieval_output_dir = os.path.join(args.output_path, "3_fewshot_retrieval")
    os.makedirs(fewshot_retrieval_output_dir, exist_ok=True)

    fewshot_retrieval_results: dict = {}
    for file in os.listdir(fewshot_retrieval_output_dir):
        if os.path.basename(file).startswith("fewshot_qid-") and file.endswith(".json"):
            question_id = int(file.split(".")[0].split("-")[-1])
            if question_id in test_question_ids:
                with open(os.path.join(fewshot_retrieval_output_dir, file), "r") as f:
                    fewshot_retrieval_results[question_id] = json.load(f)
    missing_samples = [sample for sample in test_data if sample["question_id"] not in fewshot_retrieval_results]
    logger.info(f"Loaded {len(fewshot_retrieval_results)} cached fewshot retrieval results")
    logger.info(f"Running fewshot retrieval for {len(missing_samples)} samples")
    # run fewshot retrieval for each sample
    for sample in tqdm.tqdm(test_data):
        question_id = sample["question_id"]
        if question_id in fewshot_retrieval_results:
            fewshot_retrieval_result: list[dict] = fewshot_retrieval_results[question_id]
        else:
            fewshot_retrieval_result: list[dict] = run_fewshot_retrieval(
                embedding_results[question_id].embedding, retriever, top_k=top_k
            )
        fewshot_retrieval_results[question_id] = fewshot_retrieval_result
        with open(os.path.join(fewshot_retrieval_output_dir, f"fewshot_qid-{question_id:04d}.json"), "w") as f:
            json.dump(fewshot_retrieval_result, f, indent=2)
    # assert all question ids are in fewshot_retrieval_results
    for sample in test_data:
        assert sample["question_id"] in fewshot_retrieval_results
    logger.info("Fewshot retrieval complete")

    #############################
    # sql candidate generation
    #############################
    sql_generation_output_dir = os.path.join(args.output_path, "4_candidate_generation")
    os.makedirs(sql_generation_output_dir, exist_ok=True)

    sql_generation_results: dict[str, list[CandidateGenerationOutput]] = {}
    for file in os.listdir(sql_generation_output_dir):
        if os.path.basename(file).startswith("candidate_sql_qid-") and file.endswith(".json"):
            question_id = int(file.split(".")[0].split("-")[-1])
            if question_id in test_question_ids:
                with open(os.path.join(sql_generation_output_dir, file), "r") as f:
                    bundle = CandidateGenerationOutputBundle.model_validate_json(f.read())
                    # check candidate_configs match
                    if len(bundle.candidate_configs) != len(candidate_configs):
                        logger.warning(
                            f"cached candidate_configs do not match current candidate_configs for question {question_id} (lens differ)"
                        )
                        continue
                    if any(config != candidate_configs[i] for i, config in enumerate(bundle.candidate_configs)):
                        logger.warning(
                            f"cached candidate_configs do not match current candidate_configs for question {question_id} (configs differ)"
                        )
                        continue
                    sql_generation_results[question_id] = bundle.candidates
    logger.info(f"Loaded {len(sql_generation_results)} cached sql generation results")
    missing_samples = [sample for sample in test_data if sample["question_id"] not in sql_generation_results]
    logger.info(f"Running sql generation for {len(missing_samples)} samples")
    # run sql generation for each sample, using threading executor
    with ThreadPoolExecutor(max_workers=min(2, args.num_workers)) as executor:
        futures = [
            executor.submit(
                run_candidate_sql_generation,
                sample,
                gcp_generator_candidate,
                candidate_configs,
                schema_linking_results[sample["question_id"]],
                fewshot_retrieval_results[sample["question_id"]],
            )
            for sample in missing_samples
        ]
        for future in tqdm.tqdm(futures, total=len(test_data)):
            candidates: list[CandidateGenerationOutput] = future.result()
            if len(candidates) == 0:
                logger.warning(f"No candidates found for sample {sample['question_id']}")
                continue
            question_id = candidates[0].question_id
            output = CandidateGenerationOutputBundle(
                question_id=question_id,
                candidate_configs=candidate_configs,
                candidates=candidates,
            )
            with open(os.path.join(sql_generation_output_dir, f"candidate_sql_qid-{question_id:04d}.json"), "w") as f:
                f.write(output.model_dump_json(indent=2))
            sql_generation_results[question_id] = candidates
    logger.info("Sql generation complete")

    #############################
    # rewriting
    #############################
    rewritten_results_output_dir = os.path.join(args.output_path, "5_rewritten_results")
    os.makedirs(rewritten_results_output_dir, exist_ok=True)

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
    for idx, candidate_generation_outputs in sql_generation_results.items():
        for sql_idx, candidate_generation_output in enumerate(candidate_generation_outputs):
            flattened_sqls.append((idx, sql_idx, candidate_generation_output.candidate_sql))

    # Function to run rewrite check on a single SQL
    def run_rewrite_check_wrapper(params):
        idx, sql_idx, sql = params
        sample = test_data[idx]
        try:
            output: RewriteOutput = run_rewrite_check(
                sample=sample,
                predicted_sql=sql,
                dataset=dataset,
                generator=gcp_generator_candidate,
                schema_manager=schema_manager,
            )
            return idx, sql_idx, output.rewritten_sql, output
        except Exception as e:
            logger.error(
                f"Error in rewrite check for idx {idx}, sql_idx {sql_idx}: {str(e)}: output type: {type(output).__name__}"
            )
            logger.error(f"output: {output}")
            return idx, sql_idx, sql, None  # Return original SQL if rewrite fails

    # Run rewrite check in parallel
    rewritten_sqls = {}
    if flattened_sqls:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(run_rewrite_check_wrapper, params) for params in flattened_sqls]
            for future in tqdm.tqdm(futures, total=len(flattened_sqls), desc="Running rewrite check"):
                tuple_result = future.result()
                idx, sql_idx, rewritten_sql, output = tuple_result
                if idx not in rewritten_sqls:
                    rewritten_sqls[idx] = [""] * len(sql_generation_results[idx])
                rewritten_sqls[idx][sql_idx] = rewritten_sql

    # Save rewritten results

    # Calculate and log rewrite statistics
    total_sqls = sum(len(sqls) for sqls in sql_generation_results.values())
    rewritten_sqls_count = 0

    for idx, original_sqls in sql_generation_results.items():
        rewritten_sqls_list = rewritten_sqls.get(idx, [])
        for i, original_sql in enumerate(original_sqls):
            if i < len(rewritten_sqls_list) and rewritten_sqls_list[i] != original_sql:
                rewritten_sqls_count += 1

    rewrite_percentage = (rewritten_sqls_count / total_sqls) * 100 if total_sqls > 0 else 0
    logger.info(
        f"Rewrite statistics: {rewritten_sqls_count} out of {total_sqls} SQL queries needed rewriting ({rewrite_percentage:.2f}%)"
    )
    logger.info("Rewrite check complete")

    #############################
    # candidate selection
    #############################
    candidate_selection_output_dir = os.path.join(args.output_path, "6_candidate_selection")
    os.makedirs(candidate_selection_output_dir, exist_ok=True)

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
        with open(os.path.join(candidate_selection_output_dir, f"{idx:04d}.txt"), "w") as f:
            f.write(candidate_selection_result)

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

    # todo: calculate final token counts


if __name__ == "__main__":
    main()
