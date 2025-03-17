import traceback
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from time import time

from loguru import logger
from text2sql.utils.postprocess import extract_sql_query, normalize_sql, get_table_names_from_query
from text2sql.data.schema_to_text import schema_to_datagrip_format
from text2sql.engine.prompts import BasePromptFormatter
from text2sql.pipeline.settings import GeneratorConfig

def generate_and_measure(generate_func, messages, generator_config):
    start = time()
    if generator_config:
        prediction = generate_func(messages, **generator_config)
    else:
        prediction = generate_func(messages)
    end = time()
    return prediction, end - start


def single_sample_pipe(
    test_sample: Dict,
    formatter: BasePromptFormatter,
    generator: GeneratorConfig,
    schema_description: str,
    generator_config,
    question_key: str,
    post_func,
    max_retries: int = 3,
    self_consistency: int = 1,
    embedder=None,
    retriever=None,
    use_evidence=False,
    top_k=3,
) -> Dict:
    """Process a single test sample using threads."""
    output = test_sample.copy()
    try:
        sample_query = test_sample[question_key]

        # Create chat messages
        search_results = []
        if embedder and retriever and top_k:
            search_results = retriever.query(embedder.embed(sample_query), top_k=top_k)

        generate_message_params = {
                "schema_description":schema_description,
                "query":sample_query
        }
        if search_results:
            generate_message_params.update({
                "few_shot_examples":search_results,
            })
        if use_evidence:
            generate_message_params.update({
                "evidence": test_sample["evidence"],
            })

        messages = messages = formatter.generate_messages(**generate_message_params)

        # Retry logic for generate
        last_error = None
        predictions_not_grouped = []

        # self_consistency == 1 means no self consistency, regular inference
        for _ in range(self_consistency):
            for attempt in range(max_retries):
                try:
                    prediction_all, inference_time = generate_and_measure(generator.generate, messages, generator_config)
                    prediction = post_func(prediction_all)
                    prediction = extract_sql_query(prediction)
                    prediction = normalize_sql(prediction)
                    predictions_not_grouped.append((prediction, inference_time))
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
        output["messages"] = messages
        output["llm_output"] = prediction_all
        return output

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"An error occurred before prediction: {e}\nTraceback:\n{error_traceback}")
        # Handle errors that occur before prediction
        output["error"] = str(e)
        output["predictions"] = []
        output["is_valid"] = False
        output["execution_error"] = str(e)
        return output


def repair_rewrite_pipe(
    test_sample: Dict,
    predicted_sql: str,
    formatter: BasePromptFormatter,
    generator,
    schema_description: str,
    table_text: str,
    generator_config,
    question_key: str,
    error_message: str | None = None,
    max_retries: int = 3,
) -> Dict:
    """Process a single test sample using threads."""
    try:
        sample_query = test_sample[question_key]

        if error_message:
            messages = formatter.generate_messages(
                schema_description=schema_description,
                table_text=table_text,
                query=sample_query,
                predicted_sql=predicted_sql,
                error=error_message,
            )
        else:
            messages = formatter.generate_messages(
                schema_description=schema_description,
                table_text=table_text,
                query=sample_query,
                predicted_sql=predicted_sql,
            )

        # Retry logic for generate
        prediction, last_error = None, None

        for attempt in range(max_retries):
            try:
                prediction, inference_time = generate_and_measure(generator.generate, messages, generator_config)
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
        return prediction, inference_time
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"An error occurred before prediction: {e}\nTraceback:\n{error_traceback}")
        return prediction, None


class Pipeline(ABC):
    @abstractmethod
    def run(self, test_sample):
        pass


class ConsistencyPipeline(Pipeline):
    def __init__(
        self,
        formatter: BasePromptFormatter,
        generator,
        schema_description: str | dict,
        generator_config,
        max_retries,
        self_consistency,
        embedder,
        retriever,
        top_k,
        db_instance,
        db_name,
        question_key: str,
        post_func,
        repair_formatter: BasePromptFormatter | None = None,
        rewrite_formatter: BasePromptFormatter | None = None,
        db_name_key: str | None = None,
        evidence: bool = False,
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
        self.question_key = question_key
        self.repair_formatter = repair_formatter
        self.rewrite_formatter = rewrite_formatter

        self.schema = db_instance.get_database_schema(db_name)
        self.db_name_key = db_name_key
        self.post_func = post_func

        self.evidence = evidence

    def _get_filtered_schema_description(self, prediction):
        table_names = get_table_names_from_query(prediction)
        filtered_schema = {"tables": {}}
        for table_name in table_names:
            table_name = table_name.lower()
            if table_name in self.schema["tables"]:
                filtered_schema["tables"][table_name] = self.schema["tables"][table_name]
        return schema_to_datagrip_format(self.db_name, filtered_schema)

    def validate_and_update_inferences(self, inference_result: Dict, test_sample: Dict, db_name: str) -> Dict:
        """
        Validates each prediction.
        If the results are valid and a rewrite prompt formatter exists, tries to rewrite and improve the SQL prediction.
        If the results are invalid and a repair prompt formatter exists, tries to repair the SQL prediction.
        """
        predictions_new = []
        for prediction, inference_time in inference_result["predictions"]:
            results = self.db_instance.validate_query(db_name, prediction)
            if self.rewrite_formatter or self.repair_formatter:
                filtered_schema_description = self._get_filtered_schema_description(prediction)

            obj = {
                "sql": prediction,
                "valid": False,
                "repaired": False,
                "rewritten": False,
                "inference_time_secs": inference_time,
            }
            if results.get("validated"):
                if self.rewrite_formatter:
                    repaired_prediction, rewrite_inference_time = self.run_rewrite(
                        test_sample, prediction, filtered_schema_description
                    )
                    repaired_results = self.db_instance.validate_query(db_name, prediction)
                    if repaired_results.get("validated"):
                        results = repaired_results
                results = self.db_instance.normalize_db_query_results(results)
                obj.update({"valid": True, "results": results["execution_result"]})
                if self.rewrite_formatter and repaired_results.get("validated"):
                    obj.update({"rewrite_inference_time_secs": rewrite_inference_time, "rewritten": True})
            else:
                if self.repair_formatter:
                    error_message = results.get("message")
                    repaired_prediction, repair_inference_time = self.run_repair(
                        test_sample, prediction, error_message, filtered_schema_description
                    )
                    if repaired_prediction:
                        repaired_results = self.db_instance.validate_query(db_name, repaired_prediction)
                    if repaired_prediction and repaired_results.get("validated"):
                        repaired_results = self.db_instance.normalize_db_query_results(repaired_results)
                        obj.update(
                            {
                                "sql": repaired_prediction,
                                "valid": True,
                                "repaired": True,
                                "repair_inference_time_secs": repair_inference_time,
                                "results": repaired_results["execution_result"],
                            }
                        )
                    else:
                        obj.update(
                            {
                                "repaired": True,
                            }
                        )
            predictions_new.append(obj)
        inference_result["predictions"] = predictions_new
        return inference_result

    def run_repair(
        self, test_sample: Dict, prediction: str, error_message: str, filtered_schema_description: str
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Attempts to repair a given SQL query based on the error message.
        """
        repaired_prediction, inference_time = repair_rewrite_pipe(
            test_sample,
            prediction,
            self.repair_formatter,
            self.generator,
            self.schema_description,
            filtered_schema_description,
            self.generator_config,
            self.question_key,
            error_message,
            self.max_retries,
        )
        return repaired_prediction, inference_time

    def run_rewrite(
        self, test_sample: Dict, prediction: str, filtered_schema_description: str
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Attempts to rewrite a given SQL query.
        """
        rewritten_prediction, inference_time = repair_rewrite_pipe(
            test_sample,
            prediction,
            self.rewrite_formatter,
            self.generator,
            self.schema_description,
            filtered_schema_description,
            self.generator_config,
            self.question_key,
            max_retries=self.max_retries,
        )
        return rewritten_prediction, inference_time

    def run(self, test_sample: Dict) -> Dict:
        """
        Runs the full pipeline:
        generating predictions for a sample, validating, attempting to update by rewrite or repair.
        """
        if isinstance(self.schema_description, dict):
            if self.db_name_key:
                db_name = test_sample[self.db_name_key]
                schema_description = self.schema_description[db_name]
            else:
                raise Exception(
                    "Pipeline attribute schema_description is a dict but a db_name_key is not provided."
                )
        else:
            db_name = self.db_name
            schema_description = self.schema_description
        
        schema_description = test_sample["predicted_filtered_schema_txt"] # Here is where I manually override the schema

        inference_result = single_sample_pipe(
            test_sample,
            self.formatter,
            self.generator,
            schema_description,
            self.generator_config,
            self.question_key,
            self.post_func,
            self.max_retries,
            self.self_consistency,
            self.embedder,
            self.retriever,
            use_evidence=self.evidence,
            top_k=self.top_k,
        )
        return self.validate_and_update_inferences(inference_result, test_sample, db_name)
