from typing import Dict, Any

from sqlalchemy.sql import text



def get_postgresql_schema(engine, database: str) -> Dict[str, Any]:
    """get postgres schema, columns, relations as a dictionary"""
    schema = {"tables": {}}
    with engine.begin() as connection:
        # Get table names
        # print("Getting table names")
        tables = connection.execute(
            text(
                """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """
            )
        )
        for table in tables:
            # print(f"Getting table info for '{table}'")
            table_name = table[0]
            schema["tables"][table_name] = {"columns": {}, "keys": {}, "foreign_keys": {}}

            # Get column information
            # print(f"    Getting columns for '{table_name}'")
            columns = connection.execute(
                text(
                    """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = :table_name
            """
                ),
                {"table_name": table_name},
            )
            for column in columns:
                col_name, col_type = column
                schema["tables"][table_name]["columns"][col_name] = col_type

            # Get primary key information
            # print(f"    Getting pks for '{table_name}'")
            primary_keys = connection.execute(
                text(
                    f"""
                SELECT a.attname
                FROM   pg_index i
                JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                    AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = '{table_name}'::regclass
                AND    i.indisprimary
            """
                )
            )
            pk_list = [pk[0] for pk in primary_keys]
            if pk_list:
                schema["tables"][table_name]["keys"]["primary_key"] = pk_list

            # Get foreign key information
            # print(f"getting foreign keys for '{table_name}'")
            foreign_keys = connection.execute(
                text(
                    """
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = :table_name
            """
                ),
                {"table_name": table_name},
            )
            for fk in foreign_keys:
                col_name, ref_table, ref_col = fk
                schema["tables"][table_name]["foreign_keys"][col_name] = {
                    "referenced_table": ref_table,
                    "referenced_column": ref_col,
                }
        connection.close()

    return schema