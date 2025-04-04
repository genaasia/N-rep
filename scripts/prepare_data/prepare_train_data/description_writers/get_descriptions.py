import json
import os

from get_mschema import SchemaEngine
from mac_sql.core.agents import Selector
from sqlalchemy import create_engine
from text2sql.data.datasets import SqliteDataset

M_SCHEMA_DESCS = "../data/train_descriptions/m_schema"
DAIL_AUG_DESCS = "./dail_aug_descriptions"
MAC_DESCS = "../data/train_descriptions/mac_schema"

DB_PATH = "/Users/deni/Code/synth_data/text2sql_testbed/scripts/run_pipeline/data/dev_20240627_bird/dev_databases"
DB_PATH = "/Users/deni/Code/synth_data/text2sql_testbed/scripts/table_recall/data/train_databases"
DB_PATH = "/Users/deni/Code/synth_data/text2sql_testbed/scripts/table_recall/data/train_databases"
DB_NAMES = [
    "california_schools",
    "codebase_community",
    "european_football_2",
    "formula_1",
    "superhero",
    "toxicology",
    "card_games",
    "debit_card_specializing",
    "financial",
    "student_club",
    "thrombosis_prediction",
]
DB_NAMES = [
    "beer_factory",
    "craftbeer",
    "human_resources",
    "shakespeare",
    "simpson_episodes",
    "trains",
    "video_games",
]
DB_NAMES = [
    # "beer_factory",
    "books",
    "chicago_crime",
]


def get_mschema_description(db_name):
    abs_path = os.path.join(DB_PATH, db_name, f"{db_name}.sqlite")
    db_engine = create_engine(f"sqlite:///{abs_path}")
    schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
    mschema = schema_engine.mschema
    mschema_str = mschema.to_mschema()

    with open(os.path.join(M_SCHEMA_DESCS, f"{db_name}.txt"), "w") as f:
        f.write(mschema_str)


def get_dail_aug_description(sql_dataset, db_name):
    dail_aug_str = sql_dataset.describe_database_schema(
        db_name, "basic_types_relations"
    )
    with open(os.path.join(DAIL_AUG_DESCS, f"{db_name}.txt"), "w") as f:
        f.write(dail_aug_str)


def get_mac_description(selector, db_name):
    mac_schema = selector._get_db_desc_str(db_name)[0]
    with open(os.path.join(MAC_DESCS, f"{db_name}.txt"), "w") as f:
        f.write(mac_schema)


def main():
    # sql_dataset = SqliteDataset(DB_PATH)
    selector = Selector(DB_PATH, tables_json_path="tables.json", dataset_name="bird")
    train_fn = "../data/train.json"
    with open(train_fn, "r") as f:
        train_data = json.load(f)
    train_db_ids = {row["db_id"] for row in train_data}
    train_db_ids = DB_NAMES

    # selector = Selector(
    #     DB_PATH,
    #     tables_json_path="../data/train_databases/train_tables.json",
    #     dataset_name="bird",
    # )

    for db_name in train_db_ids:
        get_mschema_description(db_name)
        # get_dail_aug_description(sql_dataset, db_name)
        # get_mac_description(selector, db_name)


if __name__ == "__main__":
    main()
