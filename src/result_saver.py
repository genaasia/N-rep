import json
import os

from loguru import logger

from models import Candidate, CandidateSelection, SchemaLinkingInfo, TokenReport, TotalTokenUsage
from text2sql.engine.embeddings import EmbeddingResult


def save_predictions(
    schema_linking_results: dict,
    embedding_results: dict,
    sql_candidate_lists: dict,
    candidate_selections: dict,
    test_data: list[dict],
    output_path: str,
):
    # check idx == question_id and add \t----- bird -----\t<db_id>
    predictions = {}
    # get ordered question ids
    ordered_question_ids = sorted(list(candidate_selections.keys()))
    for question_id in ordered_question_ids:
        selection: CandidateSelection = candidate_selections[question_id]
        db_id = selection.db_id
        prediction = selection.selected_sql
        predictions[str(question_id)] = prediction + f"\t----- bird -----\t{db_id}"
    if len(predictions) != len(test_data):
        raise ValueError(f"predictions length ({len(predictions)}) does not match test data length ({len(test_data)})")
    with open(os.path.join(output_path, "predict.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"predictions saved to {os.path.join(output_path, 'predict.json')}")

    # calculate final token counts
    total_token_counts = TotalTokenUsage(label="total")

    # for schema linking, calculate by model name
    schema_linking_token_counts: dict[str, TotalTokenUsage] = {}
    for question_id, model_schema_linking_dict in schema_linking_results.items():
        for model_name, schema_format_dict in model_schema_linking_dict.items():
            for schema_format, schema_linking_info in schema_format_dict.items():
                assert type(schema_linking_info) == SchemaLinkingInfo
                token_usage = schema_linking_info.generator_output.tokens
                schema_linking_token_counts[model_name] = (
                    schema_linking_token_counts.get(model_name, TotalTokenUsage(label=f"schema-linking_{model_name}"))
                    + token_usage
                )
                total_token_counts += token_usage
    # for sql_generation, get the generation and rewrite token counts separately
    sql_generation_token_counts = TotalTokenUsage(label="sql_generation")
    sql_generation_rewrite_token_counts = TotalTokenUsage(label="sql_generation_rewrite")
    for question_id, candidate_list in sql_candidate_lists.items():
        for candidate in candidate_list:
            assert type(candidate) == Candidate
            sql_generation_token_counts += candidate.generator_output.tokens
            total_token_counts += candidate.generator_output.tokens
            for rewrite_info in candidate.rewrite_info:
                if rewrite_info.generator_output is not None and rewrite_info.generator_output.tokens is not None:
                    sql_generation_rewrite_token_counts += rewrite_info.generator_output.tokens
                    total_token_counts += rewrite_info.generator_output.tokens

    # for candidate selection, get the selection token counts
    candidate_selection_token_counts = TotalTokenUsage(label="candidate_selection")
    for question_id, candidate_selection in candidate_selections.items():
        if type(candidate_selection) == CandidateSelection:
            for output in candidate_selection.generator_outputs:
                if hasattr(output, "tokens") and output.tokens is not None:
                    candidate_selection_token_counts += output.tokens
                    total_token_counts += output.tokens

    embedding_calls = 0
    embedding_chars = 0
    inf_time_ms = 0
    for question_id, embedding_result in embedding_results.items():
        if type(embedding_result) == EmbeddingResult:
            embedding_calls += 1
            embedding_chars += embedding_result.input_characters
            inf_time_ms += embedding_result.inf_time_ms

    embedding_result = {
        "label": "embedding",
        "calls": embedding_calls,
        "avg_characters": embedding_chars / embedding_calls,
        "ttl_characters": embedding_chars,
        "avg_inf_time_ms": inf_time_ms / embedding_calls,
        "ttl_inf_time_ms": inf_time_ms,
    }

    token_report = TokenReport(
        total=total_token_counts,
        schema_linking=schema_linking_token_counts,
        sql_generation=sql_generation_token_counts,
        sql_generation_rewrite=sql_generation_rewrite_token_counts,
        candidate_selection=candidate_selection_token_counts,
        embedding=embedding_result,
    )
    with open(os.path.join(output_path, "token_counts.json"), "w") as f:
        f.write(token_report.model_dump_json(indent=2))
    logger.info(f"token counts saved to {os.path.join(output_path, 'token_counts.json')}")
    logger.info(f"all done! check results in {output_path}")
