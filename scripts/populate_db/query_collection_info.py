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
parser.add_argument("--weaviate-collection-name", type=str, required=True, help="weaviate collection name")
parser.add_argument("--reset-weaviate-collection", action="store_true", help="reset weaviate collection")
parser.add_argument("--normalize-embeddings", action="store_true", help="normalize embeddings")
parser.add_argument("--use-weaviate-cloud", action="store_true", help="use weaviate cloud for retrieval")
parser.add_argument("--weaviate-cluster-url", type=str, default="", help="weaviate cloud cluster url, if using cloud")
parser.add_argument("--weaviate-host", type=str, default="localhost", help="local weaviate host, default 'localhost'")
parser.add_argument("--weaviate-port", type=int, default=8081, help="local weaviate port, default '8081'")
parser.add_argument("--weaviate-grpc-port", type=int, default=50051, help="local weaviate grpc port, default '50051'")
args = parser.parse_args()

load_dotenv()

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

# print info
from pprint import pprint
print(f"retriever info:")
info_dict = retriever.get_collection_info()
pprint(info_dict)
del retriever