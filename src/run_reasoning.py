import argparse
import json
import os

from concurrent.futures import ThreadPoolExecutor

import tqdm

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from text2sql.data import BaseDataset, SqliteDataset, SchemaManager
from text2sql.engine.generation import BaseGenerator, GCPGenerator, OpenAIGenerator, GenerationResult, TokenUsage
from text2sql.engine.generation.postprocessing import extract_first_code_block


PROMPT_TEMPLATE = """{database_schema}

Write an SQLite SQL query to answer the question:

{question}
{evidence}

Give the SQL query inside a ```sql markdown code block
"""


def update_moving_average(current_avg, n, new_sample):
    return (current_avg * n + new_sample) / (n + 1)


class TotalTokenUsage(BaseModel):
    label: str = ""
    calls: int = 0
    avg_inf_time_ms: float = 0
    tokens: TokenUsage = TokenUsage(prompt_tokens=0, output_tokens=0, total_tokens=0, inf_time_ms=0)

    # allow adding to TokenUsage. add to internal tokens TokenUsage and increment calls by one
    def __add__(self, other: TokenUsage) -> "TotalTokenUsage":
        self.tokens += other
        self.calls += 1
        self.avg_inf_time_ms = update_moving_average(self.avg_inf_time_ms, self.calls, other.inf_time_ms)
        return self


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
    logger.info("Creating schema manager and generating schema descriptions, this may takes some time...")
    schema_manager = SchemaManager(dataset, table_descriptions_path=table_descriptions_path)
    return dataset, schema_manager


def inference_one(
    generator: BaseGenerator,
    schema_manager: SchemaManager,
    sample: dict,
    output_path: str,
    save_messages: bool = False,
) -> GenerationResult:

    # check output_path/generations/qid_{question_id:04d}.json exists
    # if so, load it to GenerationResult and return it
    if os.path.exists(os.path.join(output_path, f"generations/qid_{sample['question_id']:04d}.json")):
        with open(os.path.join(output_path, f"generations/qid_{sample['question_id']:04d}.json"), "r") as f:
            return GenerationResult.model_validate_json(f.read())

    database = sample["db_id"]
    question = sample["question"]
    evidence = sample["evidence"]
    schema = schema_manager.get_full_schema(database, "sql_create")

    message = PROMPT_TEMPLATE.format(database_schema=schema, question=question, evidence=evidence)
    messages = [{"role": "user", "content": message}]
    result = generator.generate(messages)
    # save messages to output_path/generations/qid_{question_id:-3d}.json if save_messages is True
    if save_messages:
        with open(os.path.join(output_path, f"generations/qid_{sample['question_id']:04d}-messages.json"), "w") as f:
            json.dump(messages, f)
    # save result to output_path/generations/qid_{question_id:-3d}.json
    with open(os.path.join(output_path, f"generations/qid_{sample['question_id']:04d}.json"), "w") as f:
        json.dump(result.model_dump(), f)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-database-path", type=str, required=True, help="Path to the test database")
    parser.add_argument("--test-json-path", type=str, required=True, help="Path to the test json file")
    parser.add_argument("--test-tables-json-path", type=str, required=True, help="Path to the test tables json file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--save-messages", action="store_true", help="Save messages")
    args = parser.parse_args()
    load_dotenv()
    # create output_path/generations if it doesn't exist
    os.makedirs(os.path.join(args.output_path, "generations"), exist_ok=True)

    if "gemini" in args.model_name:
        generator = GCPGenerator(
            model=args.model_name,
            api_key=os.getenv("GCP_KEY"),
        )
    else:
        generator = OpenAIGenerator(
            model=args.model_name,
            api_key=os.getenv("OPENAI_KEY"),
        )
    logger.info(f"Using {type(generator).__name__} as the generator")

    dataset, schema_manager = prepare_dataset_information(args.test_database_path, args.test_tables_json_path)

    with open(args.test_json_path, "r") as f:
        test_data = json.load(f)
    logger.info(f"Running inference on {len(test_data)} samples")

    results: list[GenerationResult] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                inference_one,
                generator,
                schema_manager,
                sample,
                args.output_path,
                args.save_messages,
            )
            for sample in test_data
        ]
        for idx, future in tqdm.tqdm(enumerate(futures), total=len(test_data)):
            results.append(future.result())

    total_tokens = TotalTokenUsage(label="reasoning")
    final_predictions = {}
    for idx, result in enumerate(results):
        question_id = test_data[idx]["question_id"]
        sql = extract_first_code_block(result.text)
        total_tokens += result.tokens
        final_predictions[str(question_id)] = sql

    # save predictions
    with open(os.path.join(args.output_path, "predict.json"), "w") as f:
        json.dump(final_predictions, f)
    with open(os.path.join(args.output_path, "predict_dev.json"), "w") as f:
        json.dump(final_predictions, f)

    # save total tokens
    with open(os.path.join(args.output_path, "token_counts.json"), "w") as f:
        json.dump(total_tokens.model_dump(), f)


if __name__ == "__main__":
    main()

# sample run command
"""
python run_reasoning.py \
--test-database-path /data/sql_datasets/bird/dev_20240627/dev_databases \
--test-json-path /data/sql_datasets/bird/dev_20240627/dev.json \
--test-tables-json-path /data/sql_datasets/bird/dev_20240627/dev_tables.json \
--output-path ../exp/results-dev-reasoning \
--model-name o4-mini \
--num-workers 4 \
--save-messages
"""
