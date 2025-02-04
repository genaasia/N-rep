import traceback
from abc import ABC, abstractmethod
from typing import Dict

from loguru import logger
from text2sql.utils.postprocess import extract_sql_query, normalize_sql, get_table_names_from_query
from text2sql.data.schema_to_text import schema_to_datagrip_format


def single_sample_pipe(
    test_sample: Dict,
    formatter,
    generator,
    schema_description: str,
    generator_config,
    max_retries: int = 3,
    self_consistency: int = 1,
    embedder=None,
    retriever=None,
    top_k=3,
) -> Dict:
    """Process a single test sample using threads."""
    output = test_sample.copy()
    try:
        sample_query = test_sample["nl_en_query"]  # Should come from config file

        # Create chat messages
        if embedder and retriever:
            if top_k:
                search_results = retriever.query(embedder.embed(sample_query), top_k=top_k)
            else:
                search_results = []
            messages = formatter.generate_messages(
                schema_description=schema_description,
                query=sample_query,
                few_shot_examples=search_results,
            )
        else:
            messages = formatter.generate_messages(schema_description=schema_description, query=sample_query)

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
            output["error"] = f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
            output["predictions"] = []
            output["is_valid"] = False
            output["execution_error"] = str(last_error)
            return output

        output["predictions"] = predictions_not_grouped
        return output

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"An error occurred before prediction: {e}\nTraceback:\n{error_traceback}")
        # Handle errors that occur before prediction
        output["error"] = str(e)
        output["predictions"] = None
        output["is_valid"] = False
        output["execution_error"] = str(e)
        return output


def repair_pipe(
    test_sample: Dict,
    predicted_sql: str,
    error: str,
    repair_formatter,
    generator,
    schema_description: str,
    table_text: str,
    generator_config,
    max_retries: int = 3,
) -> Dict:
    """Process a single test sample using threads."""
    try:
        sample_query = test_sample["nl_en_query"]  # Should come from config file

        messages = repair_formatter.generate_messages(
            schema_description=schema_description,
            table_text=table_text,
            query=sample_query,
            predicted_sql=predicted_sql,
            error=error,
        )

        # Retry logic for generate
        prediction, last_error = None, None

        for attempt in range(max_retries):
            try:
                if generator_config:
                    prediction = generator.generate(messages, **generator_config)
                else:
                    prediction = generator.generate(messages)
                prediction = extract_sql_query(prediction)
                prediction = normalize_sql(prediction)
                break  # Success - exit retry loop
            except Exception as e:
                print(f"{attempt=}")
                print(f"{e=}")
                last_error = e
                continue

        if last_error:
            logger.error(f"An error occured during prediction: {last_error}")
        return prediction
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"An error occurred before prediction: {e}\nTraceback:\n{error_traceback}")
        return prediction


class Pipeline(ABC):
    @abstractmethod
    def run(self, test_sample):
        pass


class ConsistencyPipeline(Pipeline):
    def __init__(
        self,
        formatter,
        generator,
        schema_description,
        generator_config,
        max_retries,
        self_consistency,
        embedder,
        retriever,
        top_k,
        db_instance,
        db_name,
        repair_formatter,
    ):
        self.formatter = formatter
        self.generator = generator
        self.schema_description = schema_description
        self.generator_config = generator_config
        self.max_retries = max_retries
        self.self_consistency = self_consistency
        self.embedder = embedder
        self.retriever = retriever
        self.top_k = top_k
        self.db_instance = db_instance
        self.db_name = db_name
        self.repair_formatter = repair_formatter

        self.schema = db_instance.get_database_schema(db_name)

    def validate_and_repair_inferences(self, inference_result, test_sample):
        predictions_new = []
        for prediction in inference_result["predictions"]:
            results = self.db_instance.validate_query(self.db_name, prediction)
            if results.get("validated"):
                results = self.db_instance.normalize_db_query_results(results)
                obj = {
                    "sql": prediction,
                    "valid": True,
                    "repaired": False,
                    "results": results["execution_result"],
                }
            else:
                error_message = results.get("message")
                table_names = get_table_names_from_query(prediction)
                filtered_schema = {"tables": {}}
                for table_name in table_names:
                    filtered_schema["tables"][table_name] = self.schema["tables"][table_name]
                filtered_schema_description = schema_to_datagrip_format(self.db_name, filtered_schema)

                repaired_prediction = self.run_repair(
                    test_sample, prediction, error_message, filtered_schema_description
                )

                if repaired_prediction:
                    repaired_results = self.db_instance.validate_query(self.db_name, repaired_prediction)
                if repaired_prediction and repaired_results.get("validated"):
                    repaired_results = self.db_instance.normalize_db_query_results(repaired_results)
                    obj = {
                        "sql": repaired_prediction,
                        "valid": True,
                        "repaired": True,
                        "results": repaired_results["execution_result"],
                    }
                else:
                    obj = {"sql": prediction, "valid": False, "repaired": True}
            predictions_new.append(obj)
        inference_result["predictions"] = predictions_new
        return inference_result

    def run_repair(self, test_sample, prediction, error_message, filtered_schema_description):
        repaired_prediction = repair_pipe(
            test_sample,
            prediction,
            error_message,
            self.repair_formatter,
            self.generator,
            self.schema_description,
            filtered_schema_description,
            self.generator_config,
            self.max_retries,
        )
        return repaired_prediction

    def run(self, test_sample):
        inference_result = single_sample_pipe(
            test_sample,
            self.formatter,
            self.generator,
            self.schema_description,
            self.generator_config,
            self.max_retries,
            self.self_consistency,
            self.embedder,
            self.retriever,
            self.top_k,
        )
        return self.validate_and_repair_inferences(inference_result, test_sample)
