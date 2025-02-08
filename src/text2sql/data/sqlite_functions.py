import os
import sqlite3

from functools import lru_cache
from typing import Any

from loguru import logger


def get_sqlite_database_file(base_dir: str, database: str) -> str:
    """get path to sqlite database file based on dataset and database name"""
    # support nested and flat directory structures
    sqlite_flat_path = os.path.join(base_dir, database + ".sqlite")
    sqlite_nested_path = os.path.join(base_dir, database, database + ".sqlite")
    for sqlite_path in [sqlite_flat_path, sqlite_nested_path]:
        if os.path.exists(sqlite_path):
            return sqlite_path
    raise FileNotFoundError(f"Database file for {database=} not found in {base_dir=}")


@lru_cache(maxsize=1024)
def query_sqlite_database(base_dir: str, database: str, sql_query: str) -> list[dict]:
    """query sqlite database and return results"""
    db_path = get_sqlite_database_file(base_dir=base_dir, database=database)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    result = cursor.execute(sql_query)
    json_result = [dict(r) for r in result.fetchall()]
    connection.close()
    return json_result


@lru_cache(maxsize=1024)
def query_sqlite_database_from_connection(connection, sql_query: str) -> list[dict]:
    """query sqlite database and return results"""
    cursor = connection.cursor()
    result = cursor.execute(sql_query)
    json_result = [dict(r) for r in result.fetchall()]
    return json_result


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
