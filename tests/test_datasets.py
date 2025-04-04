import os
import json
from pathlib import Path
import pytest

import sys
sys.path.append("src")
from text2sql import hello
assert hello.message == "hello, world!"
from text2sql.data.datasets import SqliteDataset

# Constants
BASE_DATA_PATH = "/data/sql_datasets/bird/dev_20240627/dev_databases"
TABLE_DESCRIPTIONS_PATH = "/data/sql_datasets/bird/dev_20240627/dev_tables.json"
OUTPUT_DIR = "tests/outputs"
# check if the data path is dir and descriptions path exists
assert os.path.isdir(BASE_DATA_PATH)
assert os.path.exists(TABLE_DESCRIPTIONS_PATH)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load table descriptions
with open(TABLE_DESCRIPTIONS_PATH, 'r') as f:
    table_descriptions_list = json.load(f)

def get_table_descriptions_for_db(db_id: str) -> dict:
    """Get table descriptions for a specific database"""
    for desc in table_descriptions_list:
        if desc["db_id"] == db_id:
            return desc
    return {}

@pytest.fixture
def sqlite_dataset():
    return SqliteDataset(BASE_DATA_PATH)

def test_get_databases(sqlite_dataset):
    """Test that get_databases() returns the correct list of databases"""
    # Get list of database directories
    expected_dbs = [d for d in os.listdir(BASE_DATA_PATH) 
                   if os.path.isdir(os.path.join(BASE_DATA_PATH, d))]
    
    actual_dbs = sqlite_dataset.get_databases()
    
    # Sort both lists for comparison
    assert sorted(actual_dbs) == sorted(expected_dbs)

def test_get_database_path(sqlite_dataset):
    """Test that get_database_path() returns the correct path"""
    db_name = sqlite_dataset.get_databases()[0]  # Get first database
    expected_path = os.path.join(BASE_DATA_PATH, db_name, f"{db_name}.sqlite")
    actual_path = sqlite_dataset.get_database_path(db_name)
    assert actual_path == expected_path

def test_get_database_schema(sqlite_dataset):
    """Test that get_database_schema() returns a dict with 'tables' key"""
    db_name = sqlite_dataset.get_databases()[0]  # Get first database
    schema = sqlite_dataset.get_database_schema(db_name)
    assert isinstance(schema, dict)
    assert "tables" in schema

def test_describe_database_schema_modes(sqlite_dataset):
    """Test describe_database_schema() for all modes"""
    db_name = sqlite_dataset.get_databases()[0]  # Get first database
    table_descriptions = get_table_descriptions_for_db(db_name)
    
    modes = [
        "basic",
        "basic_types",
        "basic_relations",
        "basic_types_relations",
        "datagrip",
        "sql",
        "m_schema",
        "mac_schema_basic",
        "mac_schema"
    ]
    
    for mode in modes:
        # Get schema description
        description = sqlite_dataset.describe_database_schema(
            db_name,
            mode=mode,
            table_descriptions=table_descriptions if mode == "mac_schema" else None
        )
        
        # Verify it returns a string
        assert isinstance(description, str)
        
        # Write to output file
        output_file = os.path.join(OUTPUT_DIR, f"{db_name}_{mode}.txt")
        with open(output_file, 'w') as f:
            f.write(description)
