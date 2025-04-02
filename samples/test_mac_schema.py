# import from path instead of installing
import sys
sys.path.append("../src")
from text2sql import hello
print(hello.message)

import json

from text2sql.data.datasets import SqliteDataset
from text2sql.data.schema_to_text import get_mac_schema_column_samples, schema_to_mac_schema_format


bird_train_database_path = "/data/sql_datasets/bird/train/train_databases"
bird_train_tables_info_json_path = "/data/sql_datasets/bird/train/train_tables.json"

bird_train_dataset = SqliteDataset(bird_train_database_path)

address_tables_columns: dict = bird_train_dataset.get_database_schema("address")
address_column_samples: dict = get_mac_schema_column_samples(bird_train_dataset, "address", address_tables_columns)

# test both with and without extra data
with open(bird_train_tables_info_json_path, "r") as f:
    tables_info: list[dict] = json.load(f)

# get the dict where db_id == "address"
address_tables_info = next((item for item in tables_info if item["db_id"] == "address"), None)

address_mac_schema_full: str =  schema_to_mac_schema_format("address", address_tables_columns, address_column_samples, address_tables_info)
address_mac_schema_basic: str =  schema_to_mac_schema_format("address", address_tables_columns, address_column_samples)

out_file = "address_mac_schema_full.txt"
with open(out_file, "w") as f:
    f.write(address_mac_schema_full)
print(f"Address m-schema saved to {out_file}")

out_file = "address_mac_schema_basic.txt"
with open(out_file, "w") as f:
    f.write(address_mac_schema_basic)
print(f"Address m-schema saved to {out_file}")