import sys

sys.path.append("../../src")
from text2sql import hello

assert hello.message == "hello, world!", "Something went wrong importing text2sql :("

import argparse
import json
import os
import random
from time import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Dict, List

import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from loguru import logger
from text2sql.data import PostgresDataset
from text2sql.engine.retrieval import WeaviateRetriever
from text2sql.engine.embeddings import BedrockCohereEmbedder
from text2sql.evaluation.plotter import plot_accuracy

load_dotenv()

from text2sql.pipeline import Settings, single_sample_pipe
from text2sql.pipeline.tools import get_formatter, get_generator, get_postfunc, get_schema_description
from evaluation.run_eval import run_eval


def run_pipe_on_dataset(
    process_single_sample,
    formatter,
    generator,
    schema_description,
    packative_test_data,
    generator_config=None,
    batch_size=None,
    max_workers=4,
    candidate_count=1,
    embedder=None,
    retriever=None,
    top_k=0
):
    test_results = []

    process_func = partial(
        process_single_sample,
        formatter=formatter,
        generator=generator,
        schema_description=schema_description,
        generator_config=generator_config,
        self_consistency=candidate_count,
        embedder=embedder,
        retriever=retriever,
        top_k=top_k
    )

    if not batch_size:
        batch_size = len(packative_test_data)

    logger.debug(f"Will run pipeline over {batch_size} rows!")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(
            tqdm.tqdm(
                executor.map(process_func, packative_test_data[:batch_size]),
                total=batch_size,
                desc="Processing samples",
            )
        )
        test_results.extend(futures)

    return test_results


def run_inference(eval_data, pipe_configuration, settings, db_instance, db_name, embedder=None, retriever=None):
    # CREATE PROMPT FORMATTER
    formatter = get_formatter(pipe_configuration.formatter)

    # GET POST FUNCTION
    post_func = get_postfunc(pipe_configuration.postfunc)

    # CREATE GENERATOR
    generator = get_generator(
        pipe_configuration.generator.name,
        pipe_configuration.generator.model,
        post_func,
    )

    # GET SCHEMA DESCRIPTION
    schema_description = get_schema_description(pipe_configuration.schema, db_instance)

    # RUN PIPELINE OVER DATASET
    test_results = run_pipe_on_dataset(
        single_sample_pipe,
        formatter,
        generator,
        schema_description,
        eval_data,
        batch_size=settings.batch_size,
        max_workers=settings.max_workers,
        generator_config=pipe_configuration.generator.config,
        candidate_count=pipe_configuration.candidate_count,
        embedder=embedder,
        retriever=retriever,
        top_k=pipe_configuration.generator.top_k
    )
    return get_db_results_and_normalize(test_results, db_instance, db_name)


def get_db_results_and_normalize(test_results, db_instance, db_name):
    test_results_updated = []
    for i, test_result in enumerate(test_results):
        if "predictions" not in test_result:
            logger.debug(f"Couldn't find predictions in row {i}")
            continue
        predictions_new = []
        for prediction in test_result["predictions"]:
            results = db_instance.validate_query(db_name, prediction)
            if results.get("validated"):
                results = db_instance.normalize_db_query_results(results)
                obj = {
                    "sql": prediction,
                    "valid": True,
                    "results": results["execution_result"],
                }
            else:
                obj = {"sql": prediction, "valid": False}
            # predictions_new[prediction] = obj
            predictions_new.append(obj)
        test_results[i]["predictions"] = predictions_new
        test_results_updated.append(test_results[i])
    return test_results_updated


def save_results(test_results, eval_results, file_name, settings):
    outputs_file_path = os.path.join(settings.outputs_folder, f"{file_name}.json")
    if not os.path.exists(settings.outputs_folder):
        os.mkdir(settings.outputs_folder)
        logger.debug(
            f"Folder {settings.outputs_folder} doesn't exist, creating it now!"
        )
    with open(outputs_file_path, "w") as f:
        json.dump(test_results, f, indent=2)

    results_file_path = os.path.join(settings.results_folder, f"{file_name}.json")
    if not os.path.exists(settings.results_folder):
        os.mkdir(settings.results_folder)
    logger.debug(f"Folder {settings.results_folder} doesn't exist, creating it now!")
    with open(results_file_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    if not os.path.exists(settings.plots_folder):
        os.mkdir(settings.plots_folder)
        logger.debug(f"Folder {settings.plots_folder} doesn't exist, creating it now!")

    plot_file_path = os.path.join(settings.plots_folder, f"{file_name}.png")
    transformed = {}
    for key in next(iter(eval_results.values())).keys():
        transformed[key] = [entry[key] for entry in eval_results.values()]
    plot_accuracy(
        eval_results.keys(),
        transformed.values(),
        transformed.keys(),
        plot_file_path,
    )


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--run-inference",
        "-i",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run new inferences using pipeline configurations in the configuration file",
    )
    parser.add_argument(
        "--run-eval", "-e", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--update-embeddings", "-u", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--inference-file",
        "-f",
        default=None,
        help="Path to an inferenced data file, will do eval only if provided",
    )  # output_file_as_input = "preprocessed_inputs_flash1sc20_fixed.json"
    parser.add_argument(
        "--subset",
        "-s",
        default=None,
        help="Use a subset of the data for inference and or evaluation",
    )

    args = parser.parse_args()

    # Load settings from YAML
    settings = Settings.from_yaml(args.config)

    # CONFIGURE LOGGER
    formatted_date = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    log_file = f"{formatted_date}.log"
    logs_folder = settings.log_folder
    logs_file_path = os.path.join(logs_folder, log_file)
    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)
        logger.debug(f"Folder {logs_folder} doesn't exist, creating it now!")
    logger.add(logs_file_path)

    # LOAD DB AND CHECK SANITY
    packative_dataset = PostgresDataset(
        os.environ.get("POSTGRES_HOST"),
        os.environ.get("POSTGRES_PORT"),
        os.environ.get("POSTGRES_USER"),
        os.environ.get("POSTGRES_PASSWORD"),
    )
    sanity_check = packative_dataset.query_database(
        os.environ.get("POSTGRES_DB"), "SELECT COUNT(DISTINCT bank) FROM bank_account;"
    )
    assert sanity_check == [{"count": 10}]


    retriever = WeaviateRetriever(
            host=os.environ.get("WEAVIATE_HOST"), 
            port=os.environ.get("WEAVIATE_PORT"), 
            grpc_port=os.environ.get("WEAVIATE_GRPC_PORT"), 
            collection_name=settings.collection_name
        )
    embedder = BedrockCohereEmbedder(
            region_name="us-east-1",
            model="cohere.embed-multilingual-v3",
            input_type="clustering",
            batch_size=8,
        )

    if args.update_embeddings and args.run_inference:
        packative_train_data = pd.read_csv(settings.train_file_path).to_dict(orient="records")
        # packative_train_data = [datum for datum in packative_train_data if datum["validated"]]
        
        train_queries = [example["nl_en_query"] for example in packative_train_data]
        print(f"{len(train_queries)=}")
        if not os.path.isfile(settings.train_embedding_file_path):
            print(f"generating train embeddings and saving to '{settings.train_embedding_file_path}'")
            train_embeddings = embedder.embed(train_queries, verbose=True)
            np.save(settings.train_embedding_file_path, train_embeddings)
        else:
            print(f"loading train embeddings from existing file '{settings.train_embedding_file_path}'")
            train_embeddings = np.load(settings.train_embedding_file_path)

        _ = retriever.populate_collection(
                embeddings=train_embeddings,
                data=packative_train_data,
            )

    score_cache = {}
    if args.run_inference:
        packative_test_data = pd.read_csv(settings.test_file_path).to_dict(
            orient="records"
        )
        if args.subset:
            packative_test_data = random.sample(packative_test_data, int(args.subset))
        for pipe_configuration in settings.pipe_configurations:
            logger.debug(f"Running configuration {pipe_configuration.pipe_name}")
            logger.debug(
                f"Formatter: {pipe_configuration.formatter}, Schema: {pipe_configuration.schema}, Generator: {pipe_configuration.generator}"
            )

            test_results = run_inference(
                packative_test_data,
                pipe_configuration,
                settings,
                packative_dataset,
                os.environ.get("POSTGRES_DB"),
                embedder,
                retriever
            )

            # SAVE INFERENCES
            file_name = f"{pipe_configuration.pipe_name}_{formatted_date}.json"
            inference_file_path = os.path.join(settings.inference_folder, file_name)

            if not os.path.exists(settings.inference_folder):
                os.mkdir(settings.inference_folder)
                logger.debug(
                    f"Folder {settings.inference_folder} doesn't exist, creating it now!"
                )

            with open(inference_file_path, "w") as f:
                json.dump(test_results, f, indent=2)

            if args.run_eval:
                file_name = f"{pipe_configuration.pipe_name}_{formatted_date}"

                eval_results = run_eval(test_results, test_results, score_cache)

                save_results(test_results, eval_results, file_name, settings)

    if not args.run_inference and args.run_eval:
        if not args.inference_file:
            raise Exception(
                "If run_inference argument is False you must provide an inference file to do evaluation on!"
            )
        inference_file_name = os.path.splitext(os.path.basename(args.inference_file))[0]
        file_name = f"{inference_file_name}_{formatted_date}.json"
        logger.debug(
            f"Will run evaluation only without inference using predictions from this file: {inference_file_name} !!!!"
        )
        with open(args.inference_file, "r") as f:
            test_results = json.load(f)
        if args.subset:
            test_results = random.sample(test_results, int(args.subset))

        if args.run_eval:
            file_name = f"{inference_file_name}_{formatted_date}"

            eval_results = run_eval(test_results, test_results, score_cache)

            save_results(test_results, eval_results, file_name, settings)


if __name__ == "__main__":
    main()
