try:
    from text2sql import hello
except ImportError:
    import sys
    sys.path.append("../../src")
from text2sql import hello
assert hello.message == "hello, world!", "Something went wrong importing text2sql :("

import argparse
import atexit
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
from weaviate.classes.config import Property, DataType

from text2sql.engine.embeddings import BedrockCohereEmbedder
from text2sql.engine.retrieval import WeaviateRetriever, WeaviateCloudRetriever


# parse arguments
parser = argparse.ArgumentParser(description="embed data and save embeddings")
parser.add_argument("--input-file", type=str, required=True, help="path to input file. support csv, tsv or json")
parser.add_argument("--embedding-file", type=str, required=True, help="path to npy embedding file")
parser.add_argument("--weaviate-collection-name", type=str, required=True, help="weaviate collection name")
parser.add_argument("--reset-weaviate-collection", action="store_true", help="reset weaviate collection")
parser.add_argument("--normalize-embeddings", action="store_true", help="normalize embeddings")

parser.add_argument("--nl-question", type=str, required=True, help="column/key for natural-language text question data in input file")
parser.add_argument("--sql-template", type=str, required=True, help="column/key for sql template data in input file")
parser.add_argument("--sql-target", type=str, required=True, help="column/key for target sql query data in input file")
parser.add_argument("--sql-template-id", type=str, default=None, help="column/key for sql template id data in input file")
parser.add_argument("--sql-template-topic", type=str, default=None, help="column/key for sql template topic data in input file")

parser.add_argument("--use-weaviate-cloud", action="store_true", help="use weaviate cloud for retrieval")
parser.add_argument("--weaviate-cluster-url", type=str, default="", help="weaviate cloud cluster url, if using cloud")
parser.add_argument("--weaviate-host", type=str, default="localhost", help="local weaviate host, default 'localhost'")
parser.add_argument("--weaviate-port", type=int, default=8081, help="local weaviate port, default '8081'")
parser.add_argument("--weaviate-grpc-port", type=int, default=50051, help="local weaviate grpc port, default '50051'")

# parser.add_argument("--backend", type=str, default="bedrock-cohere", help="embedding model backend, default 'bedrock-cohere'")
# parser.add_argument("--bedrock-region", type=str, default="us-west-2", help="bedrock region, default 'us-west-2'")
# parser.add_argument("--model", type=str, default="embed-multilingual-v3.0", help="embedding model name, default 'embed-multilingual-v3.0'")
# parser.add_argument("--input_type", type=str, default="clustering", help="embedding model input type, default 'clustering'")
# parser.add_argument("--batch_size", type=int, default=8, help="request batch size, default 8")
# parser.add_argument("--timeout", type=int, default=60, help="request timeout in seconds, default 60")
# parser.add_argument("--sleep_ms", type=int, default=10, help="request sleep time in ms, default 10")
args = parser.parse_args()


load_dotenv()


# basic parameter validation
# check input file is csv or json
if not args.input_file.endswith(".csv") and not args.input_file.endswith(".json") and not args.input_file.endswith(".tsv"):
    raise ValueError("input file must be csv, tsv or json")
# check input file exists
if not os.path.exists(args.input_file):
    raise FileNotFoundError(f"input file '{args.input_file}' not found")
# check embedding file exists and is npy
if not args.embedding_file.endswith(".npy"):
    raise ValueError("embedding file must be npy")
if not os.path.exists(args.embedding_file):
    raise FileNotFoundError(f"embedding file '{args.embedding_file}' not found")


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
    for key in [args.nl_question, args.sql_template, args.sql_target, args.sql_template_id, args.sql_template_topic]:
        if key is not None and key not in row:
            raise ValueError(f"required key '{key}' not found in input data row '{idx}': {json.dumps(row, indent=2)}")
print(f"data loaded {len(data)} samples with keys: {data[0].keys()}")


# load embeddings
embeddings = np.load(args.embedding_file)


# create retriever
if args.use_weaviate_cloud:
    # check 'WEAVIATE_API_KEY' is set
    if "WEAVIATE_API_KEY" not in os.environ:
        raise ValueError("using weaviate cloud; required env var WEAVIATE_API_KEY not found in environment!")
    if args.weaviate_cluster_url == "":
        raise ValueError("using weaviate cloud; required argument --weaviate-cluster-url not provided!")
    retriever = WeaviateCloudRetriever(
        cluster_url=args.weaviate_cluster_url,
        collection_name=args.weaviate_collection_name,
        auth_credentials=os.environ["WEAVIATE_API_KEY"],
    )
    print(f"created cloud weaviate client for cluster '{args.weaviate_cluster_url}'")
else:
    retriever = WeaviateRetriever(
        host=args.weaviate_host,
        port=args.weaviate_port,
        grpc_port=args.weaviate_grpc_port,
        collection_name=args.weaviate_collection_name,
    )
    print(f"created local weaviate client for host '{args.weaviate_host}'")

properties=[
    Property(name="key", data_type=DataType.TEXT),
    Property(name="value", data_type=DataType.TEXT),
    Property(name="source", data_type=DataType.TEXT),
    Property(name="schema", data_type=DataType.TEXT),
    Property(name="category", data_type=DataType.TEXT),
    Property(name="labels", data_type=DataType.TEXT_ARRAY),
    Property(name="info_id", data_type=DataType.INT),
    Property(name="score", data_type=DataType.NUMBER),
    Property(name="extra", data_type=DataType.TEXT),
]

# format data to match weaviate properties
formatted_data = []
for idx, row in enumerate(tqdm.tqdm(data)):
    info_id = row[args.sql_template_id] if args.sql_template_id is not None else 0
    category = row[args.sql_template_topic] if args.sql_template_topic is not None else "none"
    formatted_data.append({
        "key": row[args.nl_question],
        "value": row[args.sql_template],
        "source": row[args.sql_target],
        "schema": "",
        "category": category,
        "labels": [],
        "info_id": info_id,
        "score": 0.0,
        "extra": "",
    })

# insert
print(f"populating collection '{args.weaviate_collection_name}' with {len(data)} samples...")
time.sleep(0.1)
result = retriever.populate_collection(
    embeddings=embeddings.tolist(),
    data=formatted_data,
    properties=properties,
    delete_existing=args.reset_weaviate_collection,
    norm=args.normalize_embeddings,
    verbose=True,
)

print(f"collection '{args.weaviate_collection_name}' populated with {len(data)} samples:")
print(json.dumps(result, indent=2))

