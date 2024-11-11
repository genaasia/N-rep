from typing import Dict, Any

import sqlalchemy
from sqlalchemy import create_engine, text

def get_mysql_schema(engine, database: str) -> Dict[str, Any]:
    """get mysql schema, columns, relations as a dictionary"""

    # Get table names
    schema = {"tables": {}}
    with engine.begin() as connection:
        connection.execute(text(f"USE {database}"))
        tables = connection.execute(text("SHOW TABLES"))
        for table in tables:
            table_name = table[0]
            schema["tables"][table_name] = {"columns": {}, "keys": {}, "foreign_keys": {}}

            # Get column information
            columns = connection.execute(text(f"DESCRIBE {table_name}"))
            for column in columns:
                col_name, col_type = column[0], column[1]
                schema["tables"][table_name]["columns"][col_name] = col_type

            # Get primary key information
            primary_keys = connection.execute(text(f"SHOW KEYS FROM {table_name} WHERE Key_name = 'PRIMARY'"))
            if primary_keys:
                schema["tables"][table_name]["keys"]["primary_key"] = [pk[4] for pk in primary_keys]

            # Get foreign key information
            foreign_keys = connection.execute(
                text(
                    f"""
                SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = '{database}' AND TABLE_NAME = '{table_name}'
                AND REFERENCED_TABLE_NAME IS NOT NULL
            """
                )
            )
            for fk in foreign_keys:
                col_name, ref_table, ref_col = fk
                schema["tables"][table_name]["foreign_keys"][col_name] = {
                    "referenced_table": ref_table,
                    "referenced_column": ref_col,
                }
        connection.close()

    return schema