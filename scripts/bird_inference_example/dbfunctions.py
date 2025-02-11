import json
import os
import re
import sqlite3

from functools import lru_cache
from typing import Any, Literal

import openai

from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt


@lru_cache(maxsize=64)
def get_sqlite_database_file(base_dir: str, database: str) -> str:
    """get path to sqlite database file based on dataset and database name"""
    # support nested and flat directory structures
    sqlite_flat_path = os.path.join(base_dir, database + ".sqlite")
    sqlite_nested_path = os.path.join(base_dir, database, database + ".sqlite")
    for sqlite_path in [sqlite_flat_path, sqlite_nested_path]:
        if os.path.exists(sqlite_path):
            return sqlite_path
    raise FileNotFoundError(f"Database file for {database=} not found in {base_dir=}")


def query_sqlite_database(base_dir: str, database: str, sql_query: str) -> tuple[str, list[dict]]:
    """query sqlite database and return status and results"""
    db_path = get_sqlite_database_file(base_dir=base_dir, database=database)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    try:
        status = "ok"
        result = cursor.execute(sql_query)
        json_result = [dict(r) for r in result.fetchall()]
    except Exception as e:
        status = f"error - {type(e).__name__}: {str(e)}"
        json_result = []
    connection.close()
    return status, json_result


@lru_cache(maxsize=64)
def get_sqlite_schema(base_dir: str, database: str) -> dict[str, Any]:
    """get sqlite schema, columns, relations as a dictionary"""
    database_path = get_sqlite_database_file(base_dir=base_dir, database=database)
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    schema = {"tables": {}}

    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        schema["tables"][table_name] = {"columns": {}, "keys": {}, "foreign_keys": {}}

        # Get column information
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()
        for column in columns:
            cid, col_name, col_type, is_notnull, default_value, is_pk = column
            schema["tables"][table_name]["columns"][col_name] = col_type
            if is_pk:
                schema["tables"][table_name]["keys"]["primary_key"] = [col_name]

        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            _, _, ref_table, col_name, ref_col, *_ = fk
            schema["tables"][table_name]["foreign_keys"][col_name] = {
                "referenced_table": ref_table,
                "referenced_column": ref_col,
            }

    cursor.close()
    connection.close()
    return schema


def schema_to_basic_format(
    schema: dict[str, Any], include_types: bool = True, include_relations: bool = True
) -> str:
    """represent schema in basic table (column, column, ...) format (following DAIL-SQL)

    this supports optional inclusion of column types and relations
    """
    output = []

    for table_name, table_info in schema["tables"].items():
        columns = []
        for col_name, col_type in table_info["columns"].items():
            col_name = str(col_name)  # Convert to string in case it's an integer
            if include_types:
                columns.append(f"{col_name} ({col_type})")
            else:
                columns.append(col_name)

        table_line = f"table '{table_name}' with columns: {' , '.join(columns)}"
        output.append(table_line)

    if include_relations:
        output.append("\nRelations:")
        for table_name, table_info in schema["tables"].items():
            if "foreign_keys" in table_info and table_info["foreign_keys"]:
                for fk_column, fk_info in table_info["foreign_keys"].items():
                    fk_column = str(fk_column)  # Convert to string in case it's an integer
                    ref_table = fk_info["referenced_table"]
                    ref_column = fk_info["referenced_column"]
                    relation = f"{table_name}.{fk_column} -> {ref_table}.{ref_column}"
                    output.append(relation)

    return "\n".join(output)