import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-database-path",
        type=str,
        required=True,
        help="path to the test databases base directory",
    )
    parser.add_argument(
        "--test-json-path",
        type=str,
        required=True,
        help="path to the test.json file",
    )
    parser.add_argument(
        "--test-tables-json-path",
        type=str,
        required=True,
        help="path to the test_tables.json file",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default="./bird_data/valid_multi_table_queries_embeddings.npy",
        help="path to preprocessed numpy embeddings file",
    )
    parser.add_argument(
        "--embeddings-data-path",
        type=str,
        default="./bird_data/valid_multi_table_queries.json",
        help="path to preprocessed json embeddings data file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        default="../outputs",
        help="target output path",
    )
    parser.add_argument(
        "--candidate-configs-path",
        type=str,
        default="./bird_data/consistency_candidate_configs.yaml",
        help="path to the candidate configs file",
    )
    parser.add_argument(
        "--column-meaning-json-path",
        type=str,
        default=None,
        help="path to the column_meaning.json file, leave blank if not used",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="run in debug mode (do small subset of data, default is None)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="number of workers to use for inference, default is 4",
    )
    parser.add_argument(
        "--save-messages",
        action="store_true",
        default=False,
        help="save messages to separate files for debugging",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=False,
        help="skip llm test",
    )
    return parser.parse_args()
