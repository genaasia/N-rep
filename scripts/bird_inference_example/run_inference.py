import argparse
import json
import os
import re
import time

from multiprocessing.pool import ThreadPool
from typing import Any, Literal

import openai
import tqdm

from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from dbfunctions import (
    get_sqlite_schema,
    schema_to_basic_format,
)

from prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE


def extract_first_code_block(text: str) -> str:
    """extract code block contents from llm output"""
    pattern = r'```(?:sql|json|\w*)\n?(.*?)\n?```'
    matches = re.finditer(pattern, text, re.DOTALL)
    results = []
    for match in matches:
        content = match.group(1).strip()
        results.append(content)
    if len(results) == 0:
        return None
    return results[0]


@retry(wait=wait_random_exponential(min=5, max=120), stop=stop_after_attempt(10))
def inference_gpt4o(
        client: AzureOpenAI, 
        messages: list[dict], 
        deployment_name: str, 
        temperature: float) -> str:
    """inference using gpt4o family model"""
    result = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=temperature
    )
    result_text = result.choices[0].message.content
    return result_text


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def inference_o3(
        client: AzureOpenAI, 
        messages: list[dict], 
        deployment_name: str, 
        reasoning_effort: Literal["low", "medium", "high"]) -> str:
    """inference using o3 family model"""
    result = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        reasoning_effort=reasoning_effort
    )
    result_text = result.choices[0].message.content
    return result_text


def run_one_inference(row: dict) -> tuple[list[dict], str]:
    question = row["question"]
    evidence = row["evidence"]
    schema_info: dict = get_sqlite_schema(
        base_dir=args.database_directory, 
        database=row["db_id"]
    )
    schema_description: str = schema_to_basic_format(
        schema=schema_info,
        include_types=True,
        include_relations=True,
    )
    user_message = USER_MESSAGE_TEMPLATE.format(
        schema_description=schema_description,
        question=question,
        evidence=evidence
    )
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": user_message}
    ]
    if args.model_name == "gena-o3-mini":
        raw_prediction: str = inference_o3(
            client=AZURE_CLIENT, 
            messages=messages, 
            deployment_name=args.model_name, 
            reasoning_effort=args.reasoning_effort
        )
    elif args.model_name in ("gena-4o", "gena-4o-2024-08-06"):
        raw_prediction: str = inference_gpt4o(
            client=AZURE_CLIENT, 
            messages=messages, 
            deployment_name=args.model_name, 
            temperature=args.temperature
        )
    else:
        raise ValueError(f"Invalid model deployment name: {args.model_name}")
    try:
        predicted_sql = extract_first_code_block(raw_prediction)
    except Exception as e:
        predicted_sql = f"ERROR - {type(e).__name__}: {str(e)}"
    return messages, predicted_sql


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="inference data")
    parser.add_argument("--input-file", type=str, required=True, help="path to input json file")
    parser.add_argument("--database-directory", type=str, required=True, help="path base directory of databases")
    parser.add_argument("--model-name", type=str, required=True, help="azure model deployment name")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature (for gpt-4o). default: 0.0")
    parser.add_argument("--reasoning-effort", type=str, default="medium", help="reasoning level (for o3-mini). default: 'medium'")
    parser.add_argument("--threads", type=int, default=4, help="number of threads (parallel jobs) to use. default: 4")
    args = parser.parse_args()

    load_dotenv()

    # validate
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"input file not found: {args.input_file}")
    if not os.path.isdir(args.database_directory):
        raise NotADirectoryError(f"database directory not found: {args.database_directory}")
    if args.reasoning_effort not in ["low", "medium", "high"]:
        raise ValueError(f"invalid reasoning effort: {args.reasoning_effort}")


    # load dev data
    with open(args.input_file, "r") as f:
        dev_data: list[dict] = json.load(f)
    print(f"Loaded {len(dev_data)} examples from {args.input_file}")


    # create the client (CAPS because GLOBAL)
    AZURE_CLIENT = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        timeout=60,
        max_retries=5,
    )

    # display settings
    if args.model_name == "gena-o3-mini":
        print(f"Running inference using {args.model_name} model with reasoning effort: {args.reasoning_effort}")
    elif args.model_name == "gena-4o-2024-08-06":
        print(f"Running inference using {args.model_name} model with temperature: {args.temperature}")
    elif args.model_name == "gena-4o":
        print(f"Running inference using {args.model_name} model with temperature: {args.temperature}")
    else:
        raise ValueError(f"Invalid model deployment name: {args.model_name}")

    # run inference 
    print(f"Running inference on {len(dev_data)} examples using {args.model_name} model")
    time.sleep(1)
    with ThreadPool(args.threads) as pool:
        prediction_output_list: list[str] = list(tqdm.tqdm(pool.imap(run_one_inference, dev_data), total=len(dev_data)))
    print(f"Finished inference on {len(dev_data)} examples")

    message_list = [{"idx": i, "messages": r[0]} for i, r in enumerate(prediction_output_list)]
    # add predictions to data and save
    for i, row in enumerate(dev_data):
        row["predicted_SQL"] = prediction_output_list[i][1]
    output_file = f"{args.input_file.replace('.json', '')}_{args.model_name}_predictions.json"
    with open(output_file, "w") as f:
        json.dump(dev_data, f, indent=2)
    message_file = f"{args.input_file.replace('.json', '')}_{args.model_name}_messages.json"
    with open(message_file, "w") as f:
        json.dump(message_list, f, indent=2)
    print(f"Saved inputs and predictions to {output_file}")




