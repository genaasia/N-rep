try:
    from text2sql import hello
except ImportError:
    import sys
    sys.path.append("../../src")
from text2sql import hello
assert hello.message == "hello, world!", "Something went wrong importing text2sql :("

import argparse
import json
import os
import random
import re
import time

from collections import Counter, defaultdict

import httpx
import numpy as np
import pandas as pd
import tqdm
import yaml

from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

from text2sql.engine.embeddings import BedrockCohereEmbedder
from text2sql.engine.retrieval import WeaviateRetriever, WeaviateCloudRetriever


def main(args):

    # basic parameter validation
    # check input file is csv or json
    if not args.input_file.endswith(".csv") and not args.input_file.endswith(".json") and not args.input_file.endswith(".tsv"):
        raise ValueError("input file must be csv, tsv or json")
    # check input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"input file '{args.input_file}' not found")
    # check output path is directory
    if not os.path.isdir(args.output_path):
        raise ValueError("output path must be a directory")


    # load data
    if args.input_file.endswith(".json"):
        data: list[dict] = pd.read_json(args.input_file)
    elif args.input_file.endswith(".csv"):
        data: list[dict] = pd.read_csv(args.input_file).to_dict(orient="records")
    elif args.input_file.endswith(".tsv"):
        data: list[dict] = pd.read_csv(args.input_file, sep="\t").to_dict(orient="records")
    # check that all required columns/keys are present
    print("validating data against keys...")
    time.sleep(0.1)
    for idx, row in enumerate(tqdm.tqdm(data)):
        if args.nl_question not in row:
            raise ValueError(f"required key '{args.nl_question}' not found in input data row '{idx}'")
    print(f"data loaded {len(data)} samples with keys: {data[0].keys()}")


    # create embedder
    if args.backend == "bedrock-cohere":
        # check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set
        if "AWS_ACCESS_KEY_ID" not in os.environ or "AWS_SECRET_ACCESS_KEY" not in os.environ:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables for bedrock inference")
        embedder = BedrockCohereEmbedder(
            region_name=args.bedrock_region,
            model=args.model,
            input_type=args.input_type,
            batch_size=args.batch_size,
            sleep_ms=args.sleep_ms,
        )
    else:
        raise NotImplementedError(f"backend '{args.backend}' not supported")


    # embed data
    print(f"embedding {len(data)} samples...")
    time.sleep(0.1)
    embeddings = embedder.embed([row[args.nl_question] for row in data], verbose=True)
    if len(embeddings) != len(data):
        raise ValueError(f"error in embedding, number of embeddings ({len(embeddings)}) not equal to number of data ({len(data)})")

    # save embeddings
    output_file_base = os.path.basename(args.input_file).rsplit(".", 1)[0]
    output_file_base_subdir = output_file_base.lower().replace(" ", "_")
    output_file_basename = f"{args.nl_question.replace(' ', '_')}_{args.backend}_{args.model}.npy"
    output_filepath = os.path.join(args.output_path, output_file_base_subdir, output_file_basename)
    if not os.path.isdir(os.path.join(args.output_path, output_file_base_subdir)):
        os.makedirs(os.path.join(args.output_path, output_file_base_subdir))
    print(f"saving embeddings to {output_filepath}...")
    embeddings_vectors = np.array(embeddings)
    np.save(output_filepath, embeddings_vectors)
    print(f"embeddings saved to:")
    print(output_filepath)
    return output_filepath


if __name__=="__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="embed data and save embeddings")
    parser.add_argument("--input-file", type=str, required=True, help="path to input file. support csv, tsv or json")
    parser.add_argument("--output-path", type=str, required=True, help="path to output directory (filename is generated from input and settings)")
    parser.add_argument("--nl-question", type=str, required=True, help="column/key for natural-language text question data in input file")
    parser.add_argument("--backend", type=str, default="bedrock-cohere", help="embedding model backend, default 'bedrock-cohere'")
    parser.add_argument("--bedrock-region", type=str, default="us-west-2", help="bedrock region, default 'us-west-2'")
    parser.add_argument("--model", type=str, default="cohere.embed-multilingual-v3", help="embedding model name, default 'embed-multilingual-v3.0'")
    parser.add_argument("--input_type", type=str, default="clustering", help="embedding model input type, default 'clustering'")
    parser.add_argument("--batch_size", type=int, default=8, help="request batch size, default 8")
    parser.add_argument("--timeout", type=int, default=60, help="request timeout in seconds, default 60")
    parser.add_argument("--sleep_ms", type=int, default=10, help="request sleep time in ms, default 10")
    args = parser.parse_args()

    load_dotenv()

    main(args)