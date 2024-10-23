from typing import Any


def schema_to_basic_format(
    database_name: str, schema: dict[str, Any], include_types: bool = False, include_relations: bool = False
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


def schema_to_sql_create(database_name: str, schema: dict[str, Any]) -> str:
    """represent schema as an SQL CREATE query statement (following DAIL-SQL)"""
    output = [f"{database_name} CREATE messages:\n"]

    for table_name, table_info in schema["tables"].items():
        create_statement = [f"CREATE TABLE {table_name} ("]
        column_definitions = []
        constraints = []

        # Columns
        for col_name, col_type in table_info["columns"].items():
            col_name = str(col_name)  # Convert to string in case it's an integer
            column_definitions.append(f"    {col_name} {col_type}")

        # Primary Key
        if "keys" in table_info and table_info["keys"].get("primary_key"):
            pk_columns = ", ".join(str(col) for col in table_info["keys"]["primary_key"])
            constraints.append(f"    PRIMARY KEY ({pk_columns})")

        # Foreign Keys
        if "foreign_keys" in table_info:
            for fk_column, fk_info in table_info["foreign_keys"].items():
                fk_column = str(fk_column)  # Convert to string in case it's an integer
                ref_table = fk_info["referenced_table"]
                ref_column = fk_info["referenced_column"]
                constraints.append(f"    FOREIGN KEY ({fk_column}) REFERENCES {ref_table} ({ref_column})")

        # Combine all parts of the CREATE TABLE statement
        create_statement.extend(column_definitions)
        if constraints:
            create_statement.extend([","] + constraints)
        create_statement.append(");")

        # Join all lines of the CREATE TABLE statement
        output.append("\n".join(create_statement))
        output.append("")  # Add an empty line between tables

    return "\n".join(output)


def schema_to_datagrip_format(database_name: str, schema: dict[str, Any]) -> str:
    """generate a very detailed schema description similar to Datagrip"""
    output = [f"{database_name} schema:"]
    output.append("    + tables")

    for table_name, table_info in schema["tables"].items():
        output.append(f"        {table_name}: table")

        # Columns
        output.append("            + columns")
        for col_name, col_type in table_info["columns"].items():
            col_name = str(col_name)  # Convert to string in case it's an integer
            output.append(f"                {col_name}: {col_type}")

        # Keys
        if "keys" in table_info and table_info["keys"]:
            output.append("            + keys")
            for key_name, key_columns in table_info["keys"].items():
                if key_name == "primary_key":
                    key_name = f"{table_name}_pk"
                key_columns = [str(col) for col in key_columns]  # Convert all column names to strings
                output.append(f"                {key_name}: PK ({', '.join(key_columns)})")

        # Foreign Keys
        if "foreign_keys" in table_info and table_info["foreign_keys"]:
            output.append("            + foreign-keys")
            for fk_column, fk_info in table_info["foreign_keys"].items():
                fk_column = str(fk_column)  # Convert to string in case it's an integer
                ref_table = fk_info["referenced_table"]
                ref_column = fk_info["referenced_column"]
                fk_name = f"{table_name}_{fk_column}_fk"
                output.append(
                    f"                {fk_name}: foreign key ({fk_column}) -> {ref_table}[.{ref_table}_pk] ({ref_column})"
                )

        output.append("")  # Add an empty line between tables

    return "\n".join(output)