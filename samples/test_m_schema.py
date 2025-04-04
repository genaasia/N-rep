# import from path instead of installing
import sys
sys.path.append("../src")
from text2sql import hello
print(hello.message)

from text2sql.data.datasets import SqliteDataset

from text2sql.data.schema_to_text import get_m_schema_column_samples, schema_to_m_schema_format


bird_train_database_path = "/data/sql_datasets/bird/train/train_databases"

bird_train_dataset = SqliteDataset(bird_train_database_path)

address_tables_columns: dict = bird_train_dataset.get_database_schema("address")

address_column_samples: dict = get_m_schema_column_samples(bird_train_dataset, "address", address_tables_columns)

address_m_schema: str =  schema_to_m_schema_format("address", address_tables_columns, address_column_samples)

out_file = "address_m_schema.txt"
with open(out_file, "w") as f:
    f.write(address_m_schema)
print(f"Address m-schema saved to {out_file}")