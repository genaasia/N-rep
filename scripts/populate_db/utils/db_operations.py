from typing import Any, Dict, List

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def truncate_table(engine, table_name: str) -> None:
    try:
        with engine.begin() as connection:
            # Disable foreign key checks temporarily
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            # Truncate the table
            connection.execute(text(f"TRUNCATE TABLE {table_name}"))
            # Re-enable foreign key checks
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        print(f"Successfully truncated table {table_name}")
    except SQLAlchemyError as e:
        print(f"Error truncating table {table_name}: {str(e)}")
        raise


def insert_data(
    connection_string: str, table_name: str, data: List[Dict[str, Any]]
) -> List[int]:
    if not data:
        print(f"No data provided for table {table_name}")
        return []

    engine = create_engine(connection_string)
    inserted_ids = []

    try:
        # Get the column names from the first data item
        columns = list(data[0].keys())
        columns_str = ", ".join(columns)
        placeholders = ", ".join([":" + col for col in columns])

        # Create the INSERT query
        query = text(
            f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        )

        # Insert data and collect IDs
        batch_size = 1000
        with engine.begin() as connection:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                for row in batch:
                    result = connection.execute(query, row)
                    # Get the last inserted ID
                    last_id_query = text("SELECT LAST_INSERT_ID()")
                    last_id = connection.execute(last_id_query).scalar()
                    inserted_ids.append(last_id)
                print(f"Inserted batch of {len(batch)} rows into {table_name}")

        print(f"Successfully inserted all {len(data)} rows into {table_name}")
        return inserted_ids

    except SQLAlchemyError as e:
        print(f"Error inserting data into {table_name}: {str(e)}")
        raise

    finally:
        engine.dispose()
