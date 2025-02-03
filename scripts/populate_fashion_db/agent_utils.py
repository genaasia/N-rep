import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from text2sql.data.datasets import MysqlDataset
from utils import insert_data, update_data


def generate_consecutive_dates(start_date, end_date):
    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end_date - start_date).days
    if total_days < 6:
        # Calculate the step size to ensure even distribution
        step = total_days / 3
        dates = [
            start_date,
            start_date + timedelta(days=round(step)),
            start_date + timedelta(days=round(2 * step)),
            start_date + timedelta(days=total_days),
        ]
        return dates

    # Initialize list with the start date
    dates = [start_date]

    # Generate 3 additional dates
    for _ in range(3):
        # Get the last generated date
        last_date = dates[-1]

        # Calculate the maximum possible date for the next date
        # It should be either end_date or 2 days after the last date, whichever is earlier
        max_possible = min(end_date, last_date + timedelta(days=2))

        # If we can't generate a new date within constraints, break
        if last_date >= max_possible:
            break

        # Generate a random date between last_date + 1 and max_possible
        days_diff = max_possible - last_date
        # random_days = random.randint(1, days_diff)
        new_date = random_date_after_date(last_date, days_diff, exp=True)

        dates.append(new_date)

    return dates


def random_dates(start_date, end_date, n):
    """
    Generate n random dates between start_date and end_date,
    ensuring at least 6 hours between each date.

    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        n (int): Number of dates to generate

    Returns:
        list: List of n random datetime objects, sorted chronologically
    """
    # Convert string dates to datetime objects
    # start = datetime.strptime(start_date + " 00:00:00", '%Y-%m-%d %H:%M:%S')
    # end = datetime.strptime(end_date + " 23:59:59", '%Y-%m-%d %H:%M:%S')

    # Calculate the total seconds between start and end
    time_delta = end_date - start_date
    total_seconds = int(time_delta.total_seconds())

    # Generate n random dates
    random_dates = []
    attempts = 0
    max_attempts = n * 100  # Prevent infinite loops

    while len(random_dates) < n and attempts < max_attempts:
        # Get a random number of seconds to add to start_date
        random_seconds = random.randint(0, total_seconds)
        candidate_date = start_date + timedelta(seconds=random_seconds)

        # Check if the candidate date is at least 6 hours apart from all existing dates
        is_valid = True
        for existing_date in random_dates:
            time_diff = abs(candidate_date - existing_date)
            if time_diff < timedelta(hours=6):
                is_valid = False
                break

        if is_valid:
            random_dates.append(candidate_date)

        attempts += 1

    if len(random_dates) < n:
        raise ValueError(
            f"Could not generate {n} dates with 6-hour spacing. Got {len(random_dates)} dates."
        )

    # Sort dates chronologically
    return sorted(random_dates)


def get_random_row(
    dataset: MysqlDataset,
    database_name: str,
    table_name: str,
    foreign_key_name: str | None = None,
    foreign_key_value: int | None = None,
    columns: list[str] | None = None,
    random_count: int = 1,
) -> dict:
    cols = ", ".join(columns) if columns else "*"
    if isinstance(foreign_key_value, str):
        foreign_key_value = f'"{foreign_key_value}"'
    if foreign_key_name and foreign_key_value:
        query = f"""
            SELECT {cols} 
            FROM {table_name}
            WHERE {foreign_key_name} = {foreign_key_value}
            ORDER BY RAND() LIMIT {random_count}"""
    else:
        query = f"""
            SELECT {cols} 
            FROM {table_name}
            ORDER BY RAND() LIMIT {random_count}"""
    results = dataset.query_database(database_name, query)
    return results


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
        random_count: int = 1,
    ) -> Dict[str, Any]:
        return get_random_row(
            self.dataset,
            self.db_name,
            table_name,
            foreign_key_name,
            foreign_key_value,
            columns,
            random_count,
        )

    def insert(self, table_name: str, data: List[Dict[str, Any]]) -> List[int]:
        return insert_data(self.conn_string, table_name, data)

    def update(
        self, table_name: str, data: List[Dict[str, Any]], where_columns: List[str]
    ) -> List[int]:
        return update_data(self.conn_string, table_name, data, where_columns)
