import os
import json
import random
from pathlib import Path
import pytest

import sys
sys.path.append("src")
from text2sql import hello
assert hello.message == "hello, world!"
from text2sql.data.datasets import SqliteDataset
from text2sql.data.schema_manager import SchemaManager

# Constants
BASE_DATA_PATH = "/data/sql_datasets/bird/dev_20240627/dev_databases"
TABLE_DESCRIPTIONS_PATH = "/data/sql_datasets/bird/dev_20240627/dev_tables.json"
OUTPUT_DIR = "tests/outputs/schema_manager"
FULL_SCHEMA_DIR = os.path.join(OUTPUT_DIR, "full")
FILTERED_SCHEMA_DIR = os.path.join(OUTPUT_DIR, "filtered")
FILTER_DICT_DIR = os.path.join(OUTPUT_DIR, "filter_dicts")

# Create output directories
os.makedirs(FULL_SCHEMA_DIR, exist_ok=True)
os.makedirs(FILTERED_SCHEMA_DIR, exist_ok=True)
os.makedirs(FILTER_DICT_DIR, exist_ok=True)

# Load table descriptions
with open(TABLE_DESCRIPTIONS_PATH, 'r') as f:
    table_descriptions_list = json.load(f)

def get_table_descriptions_for_db(db_id: str) -> dict:
    """Get table descriptions for a specific database"""
    for desc in table_descriptions_list:
        if desc["db_id"] == db_id:
            return desc
    return {}

def generate_random_filter_dict(schema: dict, num_tables: int = None) -> dict:
    """Generate a random filter dictionary from the schema.
    
    Args:
        schema: Database schema from get_database_schema()
        num_tables: Number of tables to include (None for all)
        
    Returns:
        Dictionary mapping table names to either "keep_all" or list of column names
    """
    tables = list(schema["tables"].keys())
    if num_tables is not None:
        tables = random.sample(tables, min(num_tables, len(tables)))
    
    filter_dict = {}
    for table in tables:
        columns = list(schema["tables"][table]["columns"].keys())
        # 30% chance to keep all columns
        if random.random() < 0.3:
            filter_dict[table] = "keep_all"
        else:
            # Randomly select 1-3 columns
            num_cols = random.randint(1, min(3, len(columns)))
            selected_cols = random.sample(columns, num_cols)
            filter_dict[table] = selected_cols
    
    return filter_dict

@pytest.fixture
def sqlite_dataset():
    return SqliteDataset(BASE_DATA_PATH)

@pytest.fixture
def schema_manager(sqlite_dataset):
    return SchemaManager(
        sqlite_dataset,
        table_descriptions_path=TABLE_DESCRIPTIONS_PATH
    )

def test_schema_manager_all_databases(schema_manager, sqlite_dataset):
    """Test SchemaManager for all databases with random filter dictionaries"""
    modes = [
        "basic",
        "basic_types",
        "basic_relations",
        "basic_types_relations",
        "datagrip",
        "sql",
        "m_schema",
        "mac_schema"
    ]
    
    for db_name in sqlite_dataset.get_databases():
        # Get schema for generating filter dicts
        schema = sqlite_dataset.get_database_schema(db_name)
        
        # Generate 3 different filter dictionaries
        for i in range(3):
            filter_dict = generate_random_filter_dict(schema)
            
            # Save filter dictionary
            filter_dict_file = os.path.join(FILTER_DICT_DIR, f"{db_name}_filter_{i}.json")
            with open(filter_dict_file, 'w') as f:
                json.dump(filter_dict, f, indent=2)
            
            # Test each mode
            for mode in modes:
                try:
                    # Get and save full schema
                    full_schema = schema_manager.get_full_schema(db_name, mode)
                    full_schema_file = os.path.join(FULL_SCHEMA_DIR, f"{db_name}_{mode}.txt")
                    with open(full_schema_file, 'w') as f:
                        f.write(full_schema)
                    
                    # Get and save filtered schema
                    filtered_schema = schema_manager.get_filtered_schema(db_name, filter_dict, mode)
                    filtered_schema_file = os.path.join(
                        FILTERED_SCHEMA_DIR, 
                        f"{db_name}_{mode}_filter_{i}.txt"
                    )
                    with open(filtered_schema_file, 'w') as f:
                        f.write(filtered_schema)
                        
                except Exception as e:
                    print(f"Error processing {db_name} with mode {mode} and filter {i}: {str(e)}")
                    continue 