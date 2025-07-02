import json
import os

from loguru import logger

from models import Candidate, CandidateList, CandidateSelection, SchemaLinkingInfo, AgenticRewriteResult
from text2sql.engine.embeddings import EmbeddingResult
from text2sql.data import SchemaManager


def load_schema_linking_results(schema_linking_output_dir, test_data, schema_manager: SchemaManager):
    test_question_ids = [sample["question_id"] for sample in test_data]
    question_db_ids = {sample["question_id"]: sample["db_id"] for sample in test_data}
    schema_linking_results: dict = {}
    for file in sorted(os.listdir(schema_linking_output_dir)):
        if file.startswith("schema-linking_") and file.endswith(".json"):
            # get question id from filename
            question_id = int(file.rsplit(".", 1)[0].rsplit("-", 1)[-1])
            if question_id in test_question_ids:
                with open(os.path.join(schema_linking_output_dir, file), "r") as f:
                    schema_linking_output: SchemaLinkingInfo = SchemaLinkingInfo.model_validate_json(f.read())
                    model_name = schema_linking_output.model_name
                    schema_format = schema_linking_output.schema_format
                    if question_id not in schema_linking_results:
                        schema_linking_results[question_id] = {}
                    if model_name not in schema_linking_results[question_id]:
                        schema_linking_results[question_id][model_name] = {}

                    column_description = schema_manager.get_filtered_schema(question_db_ids[question_id], schema_linking_output.column_linking, schema_format)
                    table_description = schema_manager.get_filtered_schema(question_db_ids[question_id], schema_linking_output.table_linking, schema_format)
                    full_description = schema_manager.get_full_schema(question_db_ids[question_id], schema_format)

                    schema_linking_output.column_description = column_description
                    schema_linking_output.table_description = table_description
                    schema_linking_output.full_description = full_description

                    schema_linking_results[question_id][model_name][schema_format] = schema_linking_output
    logger.info(f"Loaded {len(schema_linking_results)} cached schema linking results")
    # check how many samples not in cache based on question_id
    missing_question_ids = set([s["question_id"] for s in test_data if s["question_id"] not in schema_linking_results])
    return schema_linking_results, missing_question_ids


def load_embedding_results(embedding_output_dir, test_data):
    test_question_ids = [sample["question_id"] for sample in test_data]
    embedding_results: dict[int, EmbeddingResult] = {}
    for file in os.listdir(embedding_output_dir):
        if os.path.basename(file).startswith("embedding_qid-") and file.endswith(".json"):
            # get id from filename
            question_id = int(file.rsplit(".", 1)[0].rsplit("-", 1)[-1])
            if question_id in test_question_ids:
                with open(os.path.join(embedding_output_dir, file), "r") as f:
                    embedding_results[question_id] = EmbeddingResult.model_validate_json(f.read())
    logger.info(f"Loaded {len(embedding_results)} cached embedding results")
    missing_samples = [sample for sample in test_data if sample["question_id"] not in embedding_results]
    return embedding_results, missing_samples


def load_fewshot_retrieval_results(fewshot_retrieval_output_dir, test_data):
    test_question_ids = [sample["question_id"] for sample in test_data]
    fewshot_retrieval_results: dict = {}
    for file in os.listdir(fewshot_retrieval_output_dir):
        if os.path.basename(file).startswith("fewshot_qid-") and file.endswith(".json"):
            question_id = int(file.rsplit(".", 1)[0].rsplit("-", 1)[-1])
            if question_id in test_question_ids:
                with open(os.path.join(fewshot_retrieval_output_dir, file), "r") as f:
                    fewshot_retrieval_results[question_id] = json.load(f)
    logger.info(f"Loaded {len(fewshot_retrieval_results)} cached fewshot retrieval results")
    missing_samples = [sample for sample in test_data if sample["question_id"] not in fewshot_retrieval_results]
    return fewshot_retrieval_results, missing_samples


def load_candidate_generations(sql_generation_output_dir, test_data, candidate_configs):
    test_question_ids = [sample["question_id"] for sample in test_data]
    sql_candidate_lists: dict[int, list[Candidate]] = {}
    for file in os.listdir(sql_generation_output_dir):
        if os.path.basename(file).startswith("candidates_qid-") and file.endswith(".json"):
            question_id = int(file.rsplit(".", 1)[0].rsplit("-", 1)[-1])
            if question_id in test_question_ids:
                with open(os.path.join(sql_generation_output_dir, file), "r") as f:
                    bundle = CandidateList.model_validate_json(f.read())
                    # check candidate_configs match
                    if len(bundle.candidate_configs) != len(candidate_configs):
                        logger.warning(f"[{question_id}] cached, current candidate_configs mismatch: lens differ")
                        continue
                    if any(config != candidate_configs[i] for i, config in enumerate(bundle.candidate_configs)):
                        logger.warning(f"[{question_id}] cached, current candidate_configs mismatch: contents differ")
                        continue
                    for candidate in bundle.candidates:
                        if candidate.question_id != question_id:
                            logger.warning(
                                f"[{question_id}] cached, candidate question_id mismatch: {candidate.question_id}"
                            )
                            continue
                    sql_candidate_lists[question_id] = bundle.candidates
    logger.info(f"Loaded {len(sql_candidate_lists)} cached sql generation results")
    missing_samples = [sample for sample in test_data if sample["question_id"] not in sql_candidate_lists]
    return sql_candidate_lists, missing_samples


def load_candidate_selections(candidate_selection_output_dir, test_data):
    test_question_ids = [sample["question_id"] for sample in test_data]
    candidate_selections: dict[int, CandidateSelection] = {}
    for file in os.listdir(candidate_selection_output_dir):
        if os.path.basename(file).startswith("selection_qid-") and file.endswith(".json"):
            question_id = int(file.rsplit(".", 1)[0].rsplit("-", 1)[-1])
            if question_id in test_question_ids:
                with open(os.path.join(candidate_selection_output_dir, file), "r") as f:
                    candidate_selections[question_id] = CandidateSelection.model_validate_json(f.read())
    logger.info(f"Loaded {len(candidate_selections)} cached candidate selection results")
    missing_samples = [s for s in test_data if s["question_id"] not in candidate_selections]
    return candidate_selections, missing_samples


def load_agentic_rewrite_results(agentic_rewrite_output_dir, test_data):
    test_question_ids = [sample["question_id"] for sample in test_data]
    agentic_rewrite_results: dict[int, AgenticRewriteResult] = {}
    for file in os.listdir(agentic_rewrite_output_dir):
        if os.path.basename(file).startswith("agentic_rewrite_qid-") and file.endswith(".json"):
            question_id = int(file.rsplit(".", 1)[0].rsplit("-", 1)[-1])
            if question_id in test_question_ids:
                with open(os.path.join(agentic_rewrite_output_dir, file), "r") as f:
                    agentic_rewrite_results[question_id] = AgenticRewriteResult.model_validate_json(f.read())
    logger.info(f"Loaded {len(agentic_rewrite_results)} cached agentic rewrite results")
    missing_samples = [sample for sample in test_data if sample["question_id"] not in agentic_rewrite_results]
    return agentic_rewrite_results, missing_samples
