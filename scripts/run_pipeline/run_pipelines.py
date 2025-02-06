import sys
import copy

# sys.path.append("../../src")
from text2sql import hello

assert hello.message == "hello, world!", "Something went wrong importing text2sql :("

import argparse
import json
import os
import random
from time import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from loguru import logger
from text2sql.data import PostgresDataset, SqliteDataset, BaseDataset
from text2sql.engine.retrieval import WeaviateRetriever
from text2sql.engine.embeddings import BedrockCohereEmbedder
from text2sql.evaluation.plotter import plot_accuracy
from text2sql.engine.prompts import GenaRepairPromptFormatter, GenaRewritePromptFormatter
from text2sql.engine.generation import identity
from text2sql.pipeline.settings import PipeConfig

load_dotenv()

from text2sql.pipeline import Settings, ConsistencyPipeline
from text2sql.pipeline.tools import get_formatter, get_generator, get_postfunc, get_schema_description
from evaluation.run_eval import run_eval


def run_pipe_on_dataset(
    pipeline,
    test_data,
    batch_size=None,
    max_workers=4,
):
    test_results = []

    if not batch_size:
        batch_size = len(test_data)

    logger.debug(f"Will run pipeline over {batch_size} rows!")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(
            tqdm.tqdm(
                executor.map(pipeline.run, test_data[:batch_size]),
                total=batch_size,
                desc="Processing samples",
            )
        )
        test_results.extend(futures)

    return test_results


def run_inference(eval_data, pipe_configuration: PipeConfig, settings: Settings, db_instance: BaseDataset, db_name, database_type, embedder=None, retriever=None):
    # CREATE PROMPT FORMATTER
    formatter = get_formatter(pipe_configuration.formatter, database_type, pipe_configuration.add_date)

    # CREATE REPAIR AND REWRITE PROMPT FORMATTER
    repair_formatter = GenaRepairPromptFormatter(database_type=database_type) if pipe_configuration.repair else None
    rewrite_formatter = GenaRewritePromptFormatter(database_type=database_type) if pipe_configuration.rewrite else None

    # GET POST FUNCTION
    post_func = get_postfunc(pipe_configuration.postfunc)

    # CREATE GENERATOR
    generator = get_generator(
        pipe_configuration.generator.name,
        pipe_configuration.generator.model,
        identity,
    )

    # GET SCHEMA DESCRIPTION
    ## If the benchmark mode is on we will cache all schema descriptions in the schema_description dict
    ## If we are evaluating a single dataset like packative, schema_description will be a string
    if settings.benchmark:
        if settings.db_name_key:
            schema_description = {}
            for datum in eval_data:
                db_name = datum[settings.db_name_key]
                if db_name not in schema_description:
                    schema_description[db_name] = get_schema_description(db_name, pipe_configuration.schema, db_instance)
        else:
            raise Exception(
                "Pipeline attribute schema_description is a dict but a db_name_key is not provided."
            )
    else:
        schema_description = get_schema_description(os.environ.get("POSTGRES_DB"), pipe_configuration.schema, db_instance)


    # CREATE PIPELINE
    max_retry = 3
    pipeline = ConsistencyPipeline(
        formatter,
        generator,
        schema_description,
        pipe_configuration.generator.config,
        max_retry,
        pipe_configuration.candidate_count,
        embedder,
        retriever,
        pipe_configuration.generator.top_k,
        db_instance,
        db_name,
        settings.question_key,
        post_func,
        repair_formatter,
        rewrite_formatter,
        settings.db_name_key,
    )

    # RUN PIPELINE OVER DATASET
    test_results = run_pipe_on_dataset(
        pipeline,
        eval_data,
        batch_size=settings.batch_size,
        max_workers=settings.max_workers,
    )
    return test_results


def get_experiment_format(test_results):
    experiment_results = []
    for test_result in test_results:
        experiment_result = copy.deepcopy(test_result)
        del experiment_result["api_execution_result"]
        del experiment_result["predictions"]
        del experiment_result["messages"]
        del experiment_result["llm_output"]
        flat_keys = ["predicted_sql", "sql_match_score", "execution_match_score", "intent_score", "soft_f1_score"]
        for key in flat_keys:
            experiment_result[key]=experiment_result["highest_voted_valid"][key]
        del experiment_result["highest_voted_valid"]
        experiment_results.append(experiment_result)
    return experiment_results


def save_results(test_results, eval_results, file_name, settings):
    experiment_results = get_experiment_format(test_results)

    outputs_file_path = os.path.join(settings.outputs_folder, f"{file_name}.json")
    experiments_file_path = os.path.join(settings.outputs_folder, f"{file_name}_experiment_format.json")
    if not os.path.exists(settings.outputs_folder):
        os.mkdir(settings.outputs_folder)
        logger.debug(
            f"Folder {settings.outputs_folder} doesn't exist, creating it now!"
        )
    with open(outputs_file_path, "w") as f:
        json.dump(test_results, f, indent=2)
    with open(experiments_file_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

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

    if settings.database_type == "postgres":
        db_instance = PostgresDataset(
            os.environ.get("POSTGRES_HOST"),
            os.environ.get("POSTGRES_PORT"),
            os.environ.get("POSTGRES_USER"),
            os.environ.get("POSTGRES_PASSWORD"),
        )
    elif settings.database_type == "sqlite":
        db_instance = SqliteDataset(os.environ.get("SQLITE_DB_PATH"))
    else:
        raise Exception(
                f"Databse type {settings.database_type } is not recognized"
            ) 


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
        train_data = pd.read_csv(settings.train_file_path).to_dict(orient="records")
        # train_data = [datum for datum in train_data if datum["validated"]]
        
        train_queries = [example["nl_en_query"] for example in train_data]
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
                data=train_data,
            )

    score_cache = {}
    if args.run_inference:
        _, extension = os.path.splitext(settings.test_file_path)

        if extension == ".json":
            reader = pd.read_json
        elif extension == ".csv":
            reader = pd.read_csv
        else:
            raise Exception(
                f"Extension {extension} is not recognized for the file {settings.test_file_path}"
            )

        test_data = reader(settings.test_file_path).to_dict(
            orient="records"
        )[:settings.batch_size]
        for idx in range(len(test_data)):
            if not "api_execution_result" in test_data[idx]:
                if settings.benchmark:
                    db_name = test_data[idx][settings.db_name_key]
                else:
                    db_name = os.environ.get("POSTGRES_DB")
                result = db_instance.validate_query(db_name, test_data[idx][settings.target_sql_key])
                if result["validated"]:
                    test_data[idx]["api_execution_result"] = result["execution_result"]
                else:
                    logger.error(
                        "api_execution_result is empty and the golden query is not valid!\nSomething seems wrong with your data"
                    )
        if args.subset:
            test_data = random.sample(test_data, int(args.subset))
        for pipe_configuration in settings.pipe_configurations:
            logger.debug(f"Running configuration {pipe_configuration.pipe_name}")
            logger.debug(
                f"Formatter: {pipe_configuration.formatter}, Schema: {pipe_configuration.schema}, Generator: {pipe_configuration.generator}"
            )

            test_results = run_inference(
                test_data,
                pipe_configuration,
                settings,
                db_instance,
                os.environ.get("POSTGRES_DB"),
                settings.database_type,
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

                eval_results = run_eval(test_results, test_results, score_cache, settings.target_sql_key)

                save_results(test_results, eval_results, file_name, settings)

    retriever.client.close()

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

            eval_results = run_eval(test_results, test_results, score_cache, settings.target_sql_key)

            save_results(test_results, eval_results, file_name, settings)


if __name__ == "__main__":
    main()
