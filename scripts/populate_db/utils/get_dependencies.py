from contextlib import contextmanager
from typing import Any, Dict, List, Set

from sqlalchemy import create_engine, text


def get_table_dependencies(
    schema_text: str, target_table: str
) -> list[tuple[str, str]]:
    """
    Given a schema and a table name, returns list of tables and columns that the target table depends on.

    Args:
        schema_text (str): The full schema text containing table definitions and relations
        target_table (str): The name of the table to analyze

    Returns:
        list[tuple[str, str]]: List of tuples (referenced_table, referenced_column)
    """
    # Split schema into table definitions and relations
    parts = schema_text.split("Relations:")
    if len(parts) != 2:
        raise ValueError("Schema must contain 'Relations:' section")

    table_defs = parts[0].strip().split("\n")
    relations = parts[1].strip().split("\n")

    # Find all relations where target_table is on the left side
    dependencies = []
    for relation in relations:
        relation = relation.strip()
        if not relation:
            continue

        # Parse relation (format: Table1.column1 -> Table2.column2)
        left, right = relation.split("->")
        left = left.strip()
        right = right.strip()

        left_table, left_column = left.split(".")
        right_table, right_column = right.split(".")

        # If this is a relation for our target table
        if left_table == target_table:
            dependencies.append((right_table, right_column))

    return dependencies


class DatabaseValuesExtractor:
    def __init__(self, connection_string: str):
        """
        Initialize the extractor with database connection.

        Args:
            connection_string (str): SQLAlchemy connection string
        """
        self.engine = create_engine(connection_string)
        self.cached_values: Dict[str, Dict[str, Set[Any]]] = {}
        self.cached_all_values: Dict[str, Set[Any]] = {}

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        with self.engine.connect() as connection:
            yield connection

    def _get_column_values(self, table_name: str, column_name: str) -> Set[Any]:
        """
        Extract unique values from a specific column in a table.

        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column

        Returns:
            Set[Any]: Set of unique values from the column
        """
        # Check cache first
        if (
            table_name in self.cached_values
            and column_name in self.cached_values[table_name]
        ):
            return self.cached_values[table_name][column_name]

        # Query the database for unique values
        query = text(
            f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL"
        )

        with self.get_connection() as conn:
            result = conn.execute(query)
            values = {row[0] for row in result}

        # Cache the values
        if table_name not in self.cached_values:
            self.cached_values[table_name] = {}
        self.cached_values[table_name][column_name] = values

        return values

    def _get_all_values(self, table_name: str) -> Set[Any]:
        """
        Extract all values from a table.

        Args:
            table_name (str): Name of the table

        Returns:
            Set[Any]: Set of unique values from the column
        """
        # Check cache first
        if (
            table_name in self.cached_all_values
        ):
            return self.cached_all_values[table_name]


        # Query the database for column names
        col_query = text(
            f"SHOW COLUMNS FROM {table_name};"
        )
        with self.get_connection() as conn:
            result = conn.execute(col_query)
            col_names = [row[0] for row in result]

        # Query the database for all values
        query = text(
            f"SELECT * FROM {table_name}"
        )
        with self.get_connection() as conn:
            result = conn.execute(query)
            values = {row for row in result}

        # Cache the values
        if table_name not in self.cached_all_values:
            self.cached_all_values[table_name] = {}
        self.cached_all_values[table_name] = values, col_names

        return values, col_names

    def get_dependency_values(
        self, schema_text: str, target_table: str
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        Get all required foreign key values for a table.

        Args:
            schema_text (str): The full database schema
            target_table (str): The table to get dependencies for

        Returns:
            Dict[str, Dict[str, List[Any]]]: Dictionary of table -> column -> values mappings
        """
        # First get the dependencies
        dependencies = get_table_dependencies(schema_text, target_table)
        print(f"{dependencies=}")

        # Now get the values for each dependency
        dependency_values = {}
        for dep_table, dep_column in dependencies:
            if dep_table not in dependency_values:
                dependency_values[dep_table] = {}

            values, col_names = self._get_all_values(dep_table)
            dependency_values[dep_table] = (list(values), col_names)

        return dependency_values

    def format_dependency_values(
        self, dependency_values: Dict[str, Dict[str, List[Any]]]
    ) -> str:
        """
        Format the dependency values into a human-readable string.

        Args:
            dependency_values (Dict[str, Dict[str, List[Any]]]): The dependency values to format

        Returns:
            str: Formatted string of dependency values
        """
        lines = ["Available values for foreign keys:"]
        for table, columns in dependency_values.items():
            lines.append(f"\nFrom table '{table}':")
            for column, values in columns.items():
                # Limit the number of values shown if there are too many
                displayed_values = values[:10]
                if len(values) > 10:
                    displayed_values.append("...")
                lines.append(f"- {column}: {displayed_values}")

        return "\n".join(lines)

    def get_foreign_key_prompt(self, schema_text: str, target_table: str) -> str:
        """
        Generate a complete prompt section for foreign key constraints.

        Args:
            schema_text (str): The full database schema
            target_table (str): The table to generate prompt for

        Returns:
            str: Formatted prompt section for foreign keys
        """
        dep_values = self.get_dependency_values(schema_text, target_table)
        print(f"{dep_values=}")
        if not dep_values:
            return ""

        lines = ["Reference data from related tables:"]
        for table, (rows, col_names) in dep_values.items():
            for i, row in enumerate(rows):
                row_str = f"Table {table}, Row {i} -"
                for col_name, col_value in zip(col_names, row):
                    row_str += f" {col_name}: {col_value}"
                lines.append(row_str)
        return "\n".join(lines)
