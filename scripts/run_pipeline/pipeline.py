from typing import Dict

import sqlparse
from loguru import logger


def normalize_sql(query):
    """
    Normalize SQL query by removing extra spaces and formatting consistently.
    """
    return sqlparse.format(query, reindent=True, keyword_case="upper")


def extract_sql_query(text):
    """
    Extracts SQL query from a string containing comments and query.
    Removes comments (lines starting with --) and empty lines.

    Args:
        text (str): Input text containing SQL query and comments

    Returns:
        str: Clean SQL query without comments
    """
    # Split the text into lines
    lines = text.strip().split("\n")

    # Filter out comments and empty lines
    sql_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and comment lines
        if not line or line.startswith("--"):
            continue
        sql_lines.append(line)

    # Join the remaining lines back together
    return "\n".join(sql_lines)


def single_sample_pipe(
    test_sample: Dict,
    formatter,
    generator,
    schema_description: str,
    generator_config,
    max_retries: int = 3,
    self_consistency: int = 1,
    embedder = None,
    retriever = None,
    top_k = 3
) -> Dict:
    """Process a single test sample using threads."""
    output = test_sample.copy()
    output["db_validate_error"] = False
    try:
        sample_query = test_sample["nl_en_query"]  # Should come from config file

        # Create chat messages
        if embedder and retriever:
            if top_k:
                search_results = retriever.query(embedder.embed(sample_query), top_k=top_k)
            else:
                search_results = []
            messages = formatter.generate_messages(
                schema_description=schema_description, query=sample_query, few_shot_examples=search_results
            )
        else:
            messages = formatter.generate_messages(
                schema_description=schema_description, query=sample_query
            )

        # Retry logic for generate
        last_error = None
        predictions_not_grouped = []

        # self_consistency == 1 means no self consistency, regular inference
        for _ in range(self_consistency):
            for attempt in range(max_retries):
                try:
                    if generator_config:
                        prediction = generator.generate(messages, **generator_config)
                    else:
                        prediction = generator.generate(messages)
                    prediction = extract_sql_query(prediction)
                    prediction = normalize_sql(prediction)
                    predictions_not_grouped.append(prediction)
                    break  # Success - exit retry loop
                except Exception as e:
                    print(f"{attempt=}")
                    print(f"{e=}")
                    last_error = e
                    continue

        if len(predictions_not_grouped) == 0:
            logger.error(f"An error occured during prediction: {last_error}")
            output["error"] = (
                f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
            )
            output["predictions"] = []
            output["is_valid"] = False
            output["execution_error"] = str(last_error)
            return output

        output["predictions"] = predictions_not_grouped
        return output

    except Exception as e:
        logger.error(f"An error occured before prediction: {e}")
        # Handle errors that occur before prediction
        output["error"] = str(e)
        output["predictions"] = None
        output["is_valid"] = False
        output["execution_error"] = str(e)
        return output
