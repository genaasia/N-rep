import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from typing import Literal

import tqdm
from dotenv import load_dotenv
from loguru import logger

from args import parse_args
from bird_data.schema_linking_data import SCHEMA_LINKING_EXAMPLES
from cache_loaders import (
    load_candidate_generations,
    load_candidate_selections,
    load_embedding_results,
    load_fewshot_retrieval_results,
    load_schema_linking_results,
    load_agentic_rewrite_results,
)
from models import Candidate, CandidateList, CandidateSelection, RewriteInfo, SchemaLinkingInfo
from pipe_init_steps import (
    create_output_dir,
    load_candidate_configs,
    prepare_dataset_information,
    prepare_fewshot_retriever,
    test_generators,
    verify_env_vars,
    verify_required_files,
)
from result_saver import save_predictions
from text2sql.data import BaseDataset, SchemaManager
from text2sql.data.schema_to_text import schema_to_datagrip_format
from text2sql.engine.embeddings import BaseEmbedder, BedrockCohereEmbedder, EmbeddingResult
from text2sql.engine.generation import AzureGenerator, BaseGenerator, GCPGenerator, GenerationResult
from text2sql.engine.generation.postprocessing import extract_first_code_block
from text2sql.engine.prompts.formatters import (
    GenaCoTwEvidencePromptFormatter,
    RewritePromptFormatter,
    SchemaLinkingFewShotFormatter,
)
from text2sql.engine.retrieval import LocalRetriever
from text2sql.pipeline.selection import select_best_candidate
from text2sql.utils import parse_json_from_prediction
from text2sql.utils.postprocess import get_table_names_from_query

from agent import AgenticRewrite

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
) -> SchemaLinkingInfo:
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

    return SchemaLinkingInfo(
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
) -> list[SchemaLinkingInfo]:
    """run schema linking for all candidate configs

    Args:
        sample: one sample from the test set json
        candidate_configs: list of candidate configs
        schema_manager: the schema manager
    Returns:
        schema_linking_predictions: list of SchemaLinkingInfo predictions
    """
    jobs: list[dict] = []
    job_configs = []
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
        schema_linking_predictions: list[SchemaLinkingInfo] = list(pool.imap(run_schema_linking_wrapper, jobs))

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
    config_index: int,
    few_shot_results: list[dict],
    schema_linking_outputs: SchemaLinkingInfo,
    schema_format: str,
    schema_filtering: Literal["none", "table", "column"],
) -> Candidate:
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
    few_shot_examples.reverse()
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
    return Candidate(
        question_id=sample["question_id"],
        config_index=config_index,
        sample=sample,
        schema_format=schema_format,
        schema_filtering=schema_filtering,
        messages=messages,
        generator_output=prediction_output,
        original_sql=sql_prediction,
        candidate_sql=sql_prediction,
    )


def run_candidate_sql_generations(
    sample: dict,
    generator: BaseGenerator,
    candidate_configs: list[dict],
    candidate_schema_linking_outputs: dict[str, dict[str, SchemaLinkingInfo]],
    few_shot_results: list[dict],
) -> list[Candidate]:
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
    for candidate_idx, candidate_config in enumerate(candidate_configs):
        model = candidate_config["model"]
        schema_format = candidate_config["schema_format"]
        schema_filtering = candidate_config["schema_filtering"]
        schema_linking_output: SchemaLinkingInfo = candidate_schema_linking_outputs[model][schema_format]

        gen_args.append(
            {
                "generator": generator,
                "sample": sample,
                "config_index": candidate_idx,
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
        results: list[Candidate] = list(pool.imap(run_sql_generation_wrapper, gen_args))

    return results


def run_sql_rewrite(
    generator: BaseGenerator,
    candidate: Candidate,
    schema_description: str,
) -> RewriteInfo:
    """run the sql rewriting task"""
    is_rewritten = False
    original_sql = candidate.candidate_sql
    # format messages
    message_formatter = RewritePromptFormatter(
        database_type="sqlite",
    )
    messages = message_formatter.generate_messages(
        schema_description=schema_description,
        query=candidate.sample["question"],
        predicted_sql=original_sql,
    )
    # run inference
    rewrite_output: GenerationResult = generator.generate(messages, temperature=0.0)
    raw_prediction = rewrite_output.text
    sql_prediction = extract_first_code_block(raw_prediction)
    is_rewritten = True
    if not sql_prediction:
        sql_prediction = raw_prediction
        is_rewritten = False
    return RewriteInfo(
        question_id=candidate.question_id,
        original_sql=original_sql,
        rewritten_sql=sql_prediction,
        is_rewritten=is_rewritten,
        messages=messages,
        generator_output=rewrite_output,
    )


def check_need_rewrite(execution_results: list[dict]) -> bool:
    if len(execution_results) == 0:
        return True
    else:
        has_non_none = False
        for result_row in execution_results:
            for value in result_row.values():
                if value is not None and value != "" and value != [] and value != 0 and value != 0.0:
                    has_non_none = True
                    break
            if has_non_none:
                break
        if not has_non_none:
            return True
    return False


def get_filtered_schema_description_for_rewrite(db_name: str, schema: dict, prediction: str) -> str:
    table_names = get_table_names_from_query(prediction)
    filtered_schema = {"tables": {}}
    for table_name in table_names:
        table_name = table_name.lower()
        if table_name in schema["tables"]:
            filtered_schema["tables"][table_name] = schema["tables"][table_name]
    return schema_to_datagrip_format(db_name, filtered_schema)


def run_candidate_rewrite_check(
    candidate: Candidate,
    dataset: BaseDataset,
    generator: BaseGenerator,
    schema_manager: SchemaManager,
) -> Candidate:
    database = candidate.sample["db_id"]
    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        # Check current SQL execution
        execution_result_dict: dict = dataset.validate_query(database, candidate.candidate_sql)
        execution_results: list[dict] = execution_result_dict.get("execution_result", [])

        # If no rewrite needed, return current SQL
        if not check_need_rewrite(execution_results):
            candidate.rewrite_checked = True
            return candidate

        # Get filtered schema for rewrite
        filtered_schema_description = get_filtered_schema_description_for_rewrite(
            database, schema_manager.get_schema_mapping(database), candidate.candidate_sql
        )

        try:
            # Attempt rewrite
            rewritten_output: RewriteInfo = run_sql_rewrite(generator, candidate, filtered_schema_description)
            # add rewrite iteration info to candidate and update current sql
            candidate.candidate_sql = rewritten_output.rewritten_sql
            candidate.rewrite_info.append(rewritten_output)

        except Exception as e:
            logger.error(f"Error in run_sql_rewrite attempt {attempt + 1}: {str(e)}")
            break

        attempt += 1

    candidate.rewrite_checked = True  # should this be true even if max tries exceeded?
    return candidate


def run_candidate_selection(
    candidates: list[Candidate],
    candidate_configs: list[dict],
    dataset: BaseDataset,
    schema_manager: SchemaManager,
    generator: BaseGenerator,
    chase: bool = False,
) -> CandidateSelection:
    """run the candidate selection task"""
    # for each sample, prepare the execution result dict
    question_id: int = candidates[0].question_id
    database: str = candidates[0].sample["db_id"]
    question: str = candidates[0].sample["question"]
    evidence: str = candidates[0].sample.get("evidence", "")

    sample_dicts: list[dict] = []
    for candidate in candidates:
        sql_query = candidate.candidate_sql
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
    
    if len(candidates) == 1:
        needs_agentic_rewrite = check_need_rewrite(sample_dicts[0]["results"])
        if needs_agentic_rewrite:
            print(f"NEEDING SINGLE candidate agentic rewrite for question {question_id}")

        return CandidateSelection(
            question_id=candidates[0].question_id,
            question=question,
            db_id=database,
            candidate_config=candidate_configs[0],
            selected_idx=0,
            selected_sql=candidates[0].candidate_sql,
            needs_agentic_rewrite=needs_agentic_rewrite,
        )

    # run selection
    best_sql, chase_generations, max_vote_regular, max_vote_chase = select_best_candidate(
        predictions=sample_dicts,
        schema_manager=schema_manager,
        db_id=database,
        question=question,
        evidence=evidence,
        generator=generator,
        chase=chase,
    )
    # get the (first) index of the best sql
    sqls = [candidate.candidate_sql for candidate in candidates]
    sql_index = sqls.index(best_sql)
    # get the execution result from the sample dict
    sql_execution_results = sample_dicts[sql_index]["results"]
    assert sample_dicts[sql_index]["sql"] == best_sql
    needs_agentic_rewrite = check_need_rewrite(sql_execution_results)
    if needs_agentic_rewrite:
        print(f"NEEDING agentic rewrite for question {question_id}")

    return CandidateSelection(
        question_id=question_id,
        question=question,
        db_id=database,
        generator_outputs=chase_generations,
        candidate_config=candidate_configs[sql_index],
        selected_idx=sql_index,
        selected_sql=best_sql,
        max_vote_regular=max_vote_regular,
        max_vote_chase=max_vote_chase,
        needs_agentic_rewrite=needs_agentic_rewrite,
    )


def main():
    args = parse_args()

    load_dotenv()

    logger.info("Validating environment variables...")
    verify_env_vars()

    logger.info("Validating input files...")
    verify_required_files(args)

    logger.info("Loading candidate configs...")
    candidate_configs, top_k = load_candidate_configs(args.candidate_configs_path)

    logger.info("Creating output directory...")
    create_output_dir(args.output_path, candidate_configs)

    logger.info("Loading test data...")
    with open(args.test_json_path, "r") as f:
        test_data: list[dict] = json.load(f)
    logger.info(f"Loaded {len(test_data)} test samples")

    # load column_meaning.json
    if args.column_meaning_json_path is not None:
        with open(args.column_meaning_json_path, "r") as f:
            column_meaning_json = json.load(f)
    else:
        column_meaning_json = {}

    logger.info("Creating embedder...")
    embedder = BedrockCohereEmbedder(
        model=os.getenv("AWS_MODEL_NAME"),
        region_name=os.getenv("AWS_REGION_NAME"),
        input_type=os.getenv("AWS_INPUT_TYPE"),
        sleep_ms=10,
    )

    logger.info("Creating generators...")
    gcp_generator_candidate = GCPGenerator(
        model="gemini-1.5-flash",
        api_key=os.getenv("GCP_KEY"),
    )
    gcp_generator_selection = GCPGenerator(
        model="gemini-1.5-flash",
        api_key=os.getenv("GCP_KEY"),
    )

    if not args.skip_test:
        logger.info("Verifying generators are working...")
        test_generators()

    #############################
    # preprocessing
    #############################
    dataset, schema_manager = prepare_dataset_information(args.test_database_path, args.test_tables_json_path, args.column_meaning_json_path)
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
    logger.debug(f"First 10 test question ids: {test_question_ids[:10]}")

    #############################
    # schema linking
    # outputs: list[SchemaLinkingOutput]
    #############################
    # load any existing schema linking jsons
    schema_linking_output_dir = os.path.join(args.output_path, "1_schema_linking")
    os.makedirs(schema_linking_output_dir, exist_ok=True)
    schema_linking_results, missing_question_ids = load_schema_linking_results(schema_linking_output_dir, test_data, schema_manager)

    if len(missing_question_ids) > 0:
        logger.info(f"Running schema linking for {len(missing_question_ids)} samples")
        logger.debug(f"First 10 missing question ids: {list(missing_question_ids)[:10]}")
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
            for _, future in tqdm.tqdm(zip(missing_question_ids, futures), total=len(missing_question_ids)):
                predicted_schema_linking_outputs: list[SchemaLinkingInfo] = future.result()
                question_id = predicted_schema_linking_outputs[0].question_id
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
    else:
        logger.info("Skipping schema linking, all results loaded from cache")

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
    embedding_results, missing_samples = load_embedding_results(embedding_output_dir, test_data)

    if len(missing_samples) > 0:
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
    else:
        logger.info("Skipping embedding, all results loaded from cache")

    # assert all question ids are in embedding_results
    for sample in test_data:
        assert sample["question_id"] in embedding_results
    logger.info("Embedding complete")

    #############################
    # few-shot retrieval
    #############################
    fewshot_retrieval_output_dir = os.path.join(args.output_path, "3_fewshot_retrieval")
    os.makedirs(fewshot_retrieval_output_dir, exist_ok=True)
    fewshot_retrieval_results, missing_samples = load_fewshot_retrieval_results(fewshot_retrieval_output_dir, test_data)

    if len(missing_samples) > 0:
        logger.info(f"Running fewshot retrieval for {len(missing_samples)} samples")
        # run fewshot retrieval for each sample
        for sample in tqdm.tqdm(missing_samples):
            question_id = sample["question_id"]
            if question_id in fewshot_retrieval_results:
                fewshot_retrieval_result: list[dict] = fewshot_retrieval_results[question_id]
            else:
                fewshot_retrieval_result: list[dict] = run_fewshot_retrieval(
                    embedding_results[question_id].embeddings[0], retriever, top_k=top_k
                )
            fewshot_retrieval_results[question_id] = fewshot_retrieval_result
            with open(os.path.join(fewshot_retrieval_output_dir, f"fewshot_qid-{question_id:04d}.json"), "w") as f:
                json.dump(fewshot_retrieval_result, f, indent=2)
    else:
        logger.info("Skipping fewshot retrieval, all results loaded from cache")

    # assert all question ids are in fewshot_retrieval_results
    for sample in test_data:
        assert sample["question_id"] in fewshot_retrieval_results
    logger.info("Fewshot retrieval complete")

    #############################
    # sql candidate generation
    #############################
    sql_generation_output_dir = os.path.join(args.output_path, "4_candidate_generation")
    os.makedirs(sql_generation_output_dir, exist_ok=True)
    sql_candidate_lists, missing_samples = load_candidate_generations(
        sql_generation_output_dir, test_data, candidate_configs
    )

    if len(missing_samples) > 0:
        logger.info(f"Running sql generation for {len(missing_samples)} samples")
        # run sql generation for each sample, using threading executor
        with ThreadPoolExecutor(max_workers=min(2, args.num_workers)) as executor:
            futures = [
                executor.submit(
                    run_candidate_sql_generations,
                    sample,
                    gcp_generator_candidate,
                    candidate_configs,
                    schema_linking_results[sample["question_id"]],
                    fewshot_retrieval_results[sample["question_id"]],
                )
                for sample in missing_samples
            ]
            for future in tqdm.tqdm(futures, total=len(missing_samples)):
                candidates: list[Candidate] = future.result()
                if len(candidates) == 0:
                    logger.warning(f"No candidates found for sample {sample['question_id']}")
                    continue
                question_id = candidates[0].question_id
                sql_candidate_lists[question_id] = candidates
                # validate and save while running
                for candidate in candidates:
                    assert candidate.question_id == question_id
                if len(candidates) > 1:
                    for candidate in candidates[1:]:
                        assert candidate.sample == candidates[0].sample
                output = CandidateList(
                    question_id=question_id,
                    candidate_configs=candidate_configs,
                    candidates=candidates,
                )
                with open(os.path.join(sql_generation_output_dir, f"candidates_qid-{question_id:04d}.json"), "w") as f:
                    f.write(output.model_dump_json(indent=2))
    else:
        logger.info("Skipping sql generation, all results loaded from cache")

    # check test_data sample["question_id"] in sql_candidate_lists
    for sample in test_data:
        assert sample["question_id"] in sql_candidate_lists
    # check candidate list all have same lengths
    for question_id, candidates in sql_candidate_lists.items():
        assert len(candidates) == len(candidate_configs)
    logger.info("Sql generation complete")

    #############################
    # rewriting
    #############################
    # rewriting is using same Candidates as sql generation so do not re-load data, just save updated candidates
    # Flatten the sql_generation_results dict into a list of tuples (question id, config index, sql)
    flattened_sqls = []
    already_rewritten: int = 0
    for question_id, candidate_list in sql_candidate_lists.items():
        for config_idx, candidate in enumerate(candidate_list):
            # skip if already checked
            if candidate.rewrite_checked:
                already_rewritten += 1
                continue
            candidate_copy = copy.deepcopy(candidate)
            flattened_sqls.append((question_id, config_idx, candidate_copy))
    logger.info(f"Skipping {already_rewritten} already rewritten candidates")

    if len(flattened_sqls) > 0:
        logger.info(f"Running rewrite check for {len(flattened_sqls)} candidates")

        # Function to run rewrite check on a single SQL
        def run_rewrite_check_wrapper(params) -> tuple[int, int, str, Candidate]:
            question_id, config_idx, candidate = params
            assert type(candidate) == Candidate
            try:
                rewritten_candidate: Candidate = run_candidate_rewrite_check(
                    candidate=candidate,
                    dataset=dataset,
                    generator=gcp_generator_candidate,
                    schema_manager=schema_manager,
                )
                return question_id, config_idx, rewritten_candidate.candidate_sql, rewritten_candidate
            except Exception as e:
                logger.error(f"Error in rewrite check for idx {question_id}, sql_idx {config_idx}: {str(e)}")
                logger.error(f"output: {output}")
                return (
                    question_id,
                    config_idx,
                    candidate.candidate_sql,
                    candidate,
                )  # Return original SQL if rewrite fails

        # Run rewrite check in parallel
        if flattened_sqls:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(run_rewrite_check_wrapper, params) for params in flattened_sqls]
                for future in tqdm.tqdm(futures, total=len(flattened_sqls), desc="Running rewrite check"):
                    tuple_result: tuple[int, int, str, Candidate] = future.result()
                    question_id, config_idx, sql, candidate = tuple_result
                    # update candidate in sql_candidate_lists
                    assert question_id == candidate.question_id
                    old_cand = sql_candidate_lists[question_id][config_idx]
                    assert old_cand.question_id == candidate.question_id
                    sql_candidate_lists[question_id][config_idx] = candidate
    else:
        logger.info("Skipping rewrite check, all results loaded from cache are already rewritten")

    # assert all have been rewritten
    for question_id, candidate_list in sql_candidate_lists.items():
        for candidate in candidate_list:
            assert candidate.rewrite_checked

    # (overwrite) save data
    for question_id, candidate_list in sql_candidate_lists.items():
        bundle = CandidateList(
            question_id=question_id,
            candidate_configs=candidate_configs,
            candidates=candidate_list,
        )
        with open(os.path.join(sql_generation_output_dir, f"candidates_qid-{question_id:04d}.json"), "w") as f:
            f.write(bundle.model_dump_json(indent=2))

    logger.info("Rewrite check complete")

    #############################
    # candidate selection
    #############################
    candidate_selection_output_dir = os.path.join(args.output_path, "5_candidate_selection")
    os.makedirs(candidate_selection_output_dir, exist_ok=True)
    candidate_selections, missing_samples = load_candidate_selections(candidate_selection_output_dir, test_data)

    if len(missing_samples) > 0:
        logger.info(f"Running candidate selection for {len(missing_samples)} samples")
        # run candidate selection for each sample
        for sample in tqdm.tqdm(missing_samples):
            question_id = sample["question_id"]
            candidate_list: list[Candidate] = sql_candidate_lists[question_id]
            candidate_selection_result: CandidateSelection = run_candidate_selection(
                candidates=candidate_list,
                candidate_configs=candidate_configs,
                dataset=dataset,
                schema_manager=schema_manager,
                generator=gcp_generator_selection,
                chase=True,
            )
            candidate_selections[question_id] = candidate_selection_result
    else:
        logger.info("Skipping candidate selection, all results loaded from cache")

    # assert all question ids are in candidate_selections
    for sample in test_data:
        assert sample["question_id"] in candidate_selections

    # save all
    for question_id, candidate_selection in candidate_selections.items():
        with open(os.path.join(candidate_selection_output_dir, f"selection_qid-{question_id:04d}.json"), "w") as f:
            f.write(candidate_selection.model_dump_json(indent=2))

    logger.info("Candidate selection complete")


    agentic_rewrite_output_dir = os.path.join(args.output_path, "6_agentic_rewrite")
    os.makedirs(agentic_rewrite_output_dir, exist_ok=True)
    agentic_rewrite_results, missing_samples = load_agentic_rewrite_results(agentic_rewrite_output_dir, test_data)

    agent = AgenticRewrite(schema_manager, dataset)
    for question_id, candidate_selection in candidate_selections.items():
        if candidate_selection.max_vote_regular in [0, 1]:
            candidate_selection.needs_agentic_rewrite = True
        if candidate_selection.needs_agentic_rewrite and question_id not in agentic_rewrite_results:
            print(f"Running agentic rewrite for question {question_id}")
            agentic_rewrite_result = agent.process_row(candidate_selection)
            if agentic_rewrite_result.rewritten_sql == "PARSING_FAILED":
                continue
            agentic_rewrite_results[question_id] = agentic_rewrite_result
            with open(os.path.join(agentic_rewrite_output_dir, f"agentic_rewrite_qid-{question_id:04d}.json"), "w") as f:
                f.write(agentic_rewrite_results[question_id].model_dump_json(indent=2))


    #############################
    # output saving
    #############################
    save_predictions(
        schema_linking_results,
        embedding_results,
        sql_candidate_lists,
        candidate_selections,
        agentic_rewrite_results,
        test_data,
        args.output_path,
    )


if __name__ == "__main__":
    main()
