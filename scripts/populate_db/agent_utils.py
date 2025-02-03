import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from text2sql.data.datasets import MysqlDataset

from utils import insert_data


def get_random_row(
    dataset: MysqlDataset,
    database_name: str,
    table_name: str,
    foreign_key_name: str | None = None,
    foreign_key_value: int | None = None,
    columns: list[str] | None = None,
) -> dict:
    cols = ", ".join(columns) if columns else "*"
    if foreign_key_name and foreign_key_value:
        query = f"""
            SELECT {cols} 
            FROM {table_name}
            WHERE {foreign_key_name} = {foreign_key_value}
            ORDER BY RAND() LIMIT 1"""
    else:
        query = f"""
            SELECT {cols} 
            FROM {table_name}
            ORDER BY RAND() LIMIT 1"""
    result = dataset.query_database(database_name, query)
    return result[0] if result else None


def get_related_rows(
    dataset: MysqlDataset,
    database_name: str,
    table_name: str,
    foreign_key_name: str,
    foreign_key_value: int,
    columns: list[str] | None = None,
) -> list[dict]:
    cols = ", ".join(columns) if columns else "*"
    query = f"""
        SELECT {cols} 
        FROM {table_name} 
        WHERE {foreign_key_name} = {foreign_key_value}
    """
    return dataset.query_database(database_name, query)


def random_selection(items):
    # Randomly decide how many items to select (1 to 5)
    weights = [0.5, 0.2, 0.15, 0.1, 0.05]
    num_to_select = random.choices([1, 2, 3, 4, 5], weights=weights)[0]

    # Make selections with replacement (same item can be chosen multiple times)
    selections = [random.choice(items) for _ in range(num_to_select)]

    return selections


def random_date_after_date(start_date, timedelta_range=None, exp=False):
    start_datetime = datetime.strptime(str(start_date), "%Y-%m-%d %H:%M:%S")
    if timedelta_range:
        end_date = start_datetime + timedelta_range
    else:
        end_date = datetime.now()

    # Generate random timestamp between order date and max refund date
    time_diff = end_date - start_datetime
    if exp:
        random_seconds = (
            int((random.random() ** 1.7) * int(time_diff.total_seconds())) + 60
        )
    else:
        random_seconds = random.randint(0, int(time_diff.total_seconds()))
    rand_date = start_datetime + timedelta(seconds=random_seconds)
    return rand_date


class DatabaseConnection:
    def __init__(self, dataset: MysqlDataset, db_name: str):
        self.dataset = dataset
        self.db_name = db_name
        self.conn_string = f"{self.dataset._get_connection_string()}/{self.db_name}"

    def get_related_rows(
        self,
        table_name: str,
        foreign_key_name: str,
        foreign_key_value: int,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return get_related_rows(
            self.dataset,
            self.db_name,
            table_name,
            foreign_key_name,
            foreign_key_value,
            columns,
        )

    def get_random_row(
        self,
        table_name: str,
        foreign_key_name: Optional[str] = None,
        foreign_key_value: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return get_random_row(
            self.dataset,
            self.db_name,
            table_name,
            foreign_key_name,
            foreign_key_value,
            columns,
        )

    def insert(self, table_name: str, data: List[Dict[str, Any]]) -> List[int]:
        return insert_data(self.conn_string, table_name, data)
