import json
import os
import re
import sys

from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.automap import automap_base

sys.path.append("../../src")
from dotenv import load_dotenv
from text2sql.data.datasets import MysqlDataset
from text2sql.engine.generation import AzureGenerator

from utils import (Config, DatabaseValuesExtractor, format_prompt, insert_data,
                   parse_args, truncate_table)

load_dotenv()


def get_insert_order(conn_url):
    engine = create_engine(url=conn_url)
    metadata = MetaData()
    metadata.reflect(engine)
    Base = automap_base(metadata=metadata)
    Base.prepare()
    tables_names = list(metadata.sorted_tables)
    return tables_names


def extract_first_json_block(text: str) -> str:
    """extract code block contents"""
    pattern = r"```(?:sql|python|json|\w*)\n?(.*?)\n?```"
    matches = re.finditer(pattern, text, re.DOTALL)
    results = []
    for match in matches:
        content = match.group(1).strip()
        results.append(content)
    if len(results) == 0:
        return None
    return results[0]


def main():
    args = parse_args()
    config = Config(args.config)

    ## GET DB CONNECTION
    dataset = MysqlDataset(
        os.environ.get("MYSQL_HOST"),
        os.environ.get("MYSQL_PORT"),
        os.environ.get("MYSQL_USER"),
        os.environ.get("MYSQL_PASSWORD"),
    )

    ## GET DB SCHEMA
    schema = dataset.describe_database_schema(config.db_name, "basic_types_relations")

    ## GET GENERATOR LLM
    generator = AzureGenerator(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_API_ENDPOINT"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        model=os.environ.get("AZURE_OPENAI_GEN_MODEL"),
        post_func=extract_first_json_block,
    )

    db_conn_string = f"{dataset._get_connection_string()}/{config.db_name}"

    ## GET INSERT ORDER
    insert_order = get_insert_order(db_conn_string)
    if config.subset:
        insert_order = insert_order[: config.subset]

    ## GET DEPENDENCY EXTRACTOR FOR FOREIGN KEYS
    extractor = DatabaseValuesExtractor(db_conn_string)

    ## CLEAN TABLES
    for table in insert_order:
        engine = create_engine(db_conn_string)
        truncate_table(engine, table)

    ## GENERATE DUMMY DB AND POPULATE DB
    for table in insert_order:
        file_path = os.path.join(config.data_folder, f"{table}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                prediction = json.load(f)
        else:
            fk_prompt = extractor.get_foreign_key_prompt(schema, str(table))
            prompt = format_prompt(schema, table, fk_prompt)
            messages = [{"role": "user", "content": prompt}]
            prediction = generator.generate(messages)
            prediction = json.loads(prediction)
            with open(file_path, "w") as f:
                json.dump(prediction, f, indent=2)

        insert_data(db_conn_string, table, prediction)


if __name__ == "__main__":
    main()
