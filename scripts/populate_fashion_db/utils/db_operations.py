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
        # batch_size = 1000
        # with engine.begin() as connection:
        #     for i in range(0, len(data), batch_size):
        #         batch = data[i : i + batch_size]
        #         for row in batch:
        #             result = connection.execute(query, row)
        #             # Get the last inserted ID
        #             last_id_query = text("SELECT LAST_INSERT_ID()")
        #             last_id = connection.execute(last_id_query).scalar()
        #             inserted_ids.append(last_id)
        #         # print(f"Inserted batch of {len(batch)} rows into {table_name}")
        # print(f"Successfully inserted all {len(data)} rows into {table_name}")
        # return inserted_ids

        # Insert data in batches
        batch_size = 1000
        with engine.begin() as connection:  # This automatically handles commit/rollback
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                connection.execute(query, batch)
                print(f"Inserted batch of {len(batch)} rows into {table_name}")
                last_id_query = text("SELECT LAST_INSERT_ID()")
                last_id = connection.execute(last_id_query).scalar()
                inserted_ids.append(last_id)

        print(f"Successfully inserted all {len(data)} rows into {table_name}")
        return inserted_ids
      
    except SQLAlchemyError as e:
        print(f"Error inserting data into {table_name}: {str(e)}")
        raise

    finally:
        engine.dispose()


def update_data(
    connection_string: str,
    table_name: str,
    data: List[Dict[str, Any]],
    where_columns: List[str],
) -> int:
    """
    Update rows in a database table based on specified conditions.

    Args:
        connection_string: Database connection string
        table_name: Name of the table to update
        data: List of dictionaries containing the data to update
        where_columns: List of column names to use in the WHERE clause

    Returns:
        Number of rows updated
    """
    if not data:
        print(f"No data provided for updating table {table_name}")
        return 0

    if not where_columns:
        raise ValueError("No where columns specified for update operation")

    engine = create_engine(connection_string)
    rows_updated = 0

    try:
        # Create the UPDATE query template
        set_columns = [col for col in data[0].keys() if col not in where_columns]
        set_clause = ", ".join([f"{col} = :{col}" for col in set_columns])
        where_clause = " AND ".join([f"{col} = :{col}" for col in where_columns])

        query = text(f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}")

        # Update data in batches
        batch_size = 1000
        with engine.begin() as connection:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                for row in batch:
                    result = connection.execute(query, row)
                    rows_updated += result.rowcount
                # print(f"Updated batch of {len(batch)} rows in {table_name}")

        # print(f"Successfully updated {rows_updated} rows in {table_name}")
        return rows_updated

    except SQLAlchemyError as e:
        print(f"Error updating data in {table_name}: {str(e)}")
        raise

    finally:
        engine.dispose()
