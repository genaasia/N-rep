import argparse
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

from text2sql.data import SqliteDataset
from text2sql.engine.generation import GCPGenerator

load_dotenv()
GCP_KEY = os.getenv("GCP_KEY")

TEMPERATURE = 0
MODEL_NAME = "gemini-2.5-pro"

SYSTEM_PROMPT = "You are a helpful assistant that writes detailed column descriptions for database profiling. Your goal is to generate clear, structured, and fact-based metadata summaries for each column in a SQL database. The user will give you profiling statistics and sample values for a single column. Use only the provided data and reasonable inferences based on naming patterns, common abbreviations, and context. If a column name includes a well-known acronym (e.g., CDS), try to expand it using standard domain knowledge when appropriate."
USER_PROMPT = """Given the following profiling information for a column in a relational database, generate a long-form description suitable for use in a text-to-SQL prompt.

The description should:
- Explain the meaning of the column if the name isn't descriptive enough.
- Begin with what the column stores, if evident from the name or shape.
- Mention the presence or absence of NULLs and the total number of records.
- Report the number of distinct values.
- Mention the min and max values.
- Describe the format (e.g. fixed length, numeric, alphanumeric).
- List the top 10 most common non-null values.

Make use of the information provided for the whole schema to understand a single column.

Give the description as a single paragraph"""


def load_column_meaning(column_meaning_path: str):
    with open(column_meaning_path, "r") as f:
        column_meaning = json.load(f)
    column_meaning_parsed = {}
    for key in column_meaning.keys():
        db_name, table_name, column_name = key.split("|")
        if db_name not in column_meaning_parsed:
            column_meaning_parsed[db_name] = {}
        if table_name not in column_meaning_parsed[db_name]:
            column_meaning_parsed[db_name][table_name] = {}
        column_meaning_parsed[db_name][table_name][column_name] = column_meaning[key]
    return column_meaning_parsed


def profile_table(
    db_path: str, table_name: str, top_k: int = 20, column_meaning_parsed: dict = None, wanted_cols: List[str] = None
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    db_name = db_path.split("/")[-1].split(".")[0]

    # print(f"Profiling table: {table_name}")

    # Get column names and types
    cursor.execute(f"PRAGMA table_info(`{table_name}`);")
    columns = cursor.fetchall()
    col_info = [(col[1], col[2]) for col in columns]

    # Total row count
    # cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
    # total_rows = cursor.fetchone()[0]
    # print(f"Total rows: {total_rows}\n")

    column_profiles = {}

    for col_name, col_type in col_info:
        if wanted_cols is not None and col_name not in wanted_cols:
            continue

        # Build the profile string for this column
        profile_lines = []
        profile_lines.append(f"--- Column: {col_name} ({col_type}) ---")

        if col_name in column_meaning_parsed[db_name][table_name]:
            profile_lines.append(f"  {column_meaning_parsed[db_name][table_name][col_name]}")

        # NULL / non-NULL
        cursor.execute(
            f"""
            SELECT COUNT(*) - COUNT("{col_name}"), COUNT("{col_name}")
            FROM `{table_name}`;
        """
        )
        nulls, non_nulls = cursor.fetchone()
        profile_lines.append(f"  NULLs: {nulls}, Non-NULLs: {non_nulls}")

        # Distinct count
        cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM `{table_name}`;')
        distinct_count = cursor.fetchone()[0]
        profile_lines.append(f"  Distinct values: {distinct_count}")

        # Min/max values
        cursor.execute(f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM `{table_name}`;')
        min_val, max_val = cursor.fetchone()
        profile_lines.append(f"  Min: {min_val}, Max: {max_val}")

        # Length stats for strings
        if col_type.lower() in ["text", "varchar", "char"]:
            cursor.execute(
                f"""
                SELECT MIN(LENGTH("{col_name}")), MAX(LENGTH("{col_name}"))
                FROM `{table_name}`
                WHERE "{col_name}" IS NOT NULL;
            """
            )
            min_len, max_len = cursor.fetchone()
            profile_lines.append(f"  String Length: min {min_len}, max {max_len}")

        # Top-k values
        profile_lines.append(f"  Top {top_k} most frequent values:")
        cursor.execute(
            f"""
            SELECT "{col_name}", COUNT(*) as freq
            FROM `{table_name}`
            WHERE "{col_name}" IS NOT NULL
            GROUP BY "{col_name}"
            ORDER BY freq DESC
            LIMIT {top_k};
        """
        )
        for val, freq in cursor.fetchall():
            profile_lines.append(f"    value: {val}, frequency: {freq}")

        # Store the complete profile as a string in the dictionary
        column_profiles[col_name] = "\n".join(profile_lines)

    conn.close()
    return column_profiles


def process_column(
    db_name,
    table_name,
    col_name,
    descriptions,
    dataset,
    gcp_generator_candidate,
    llm_descriptions,
    llm_descriptions_lock,
):
    """Process a single column to generate LLM description"""
    if col_name in llm_descriptions[db_name][table_name] and llm_descriptions[db_name][table_name][col_name]:
        return
    test_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT
            + "\nDatabase schema:\n"
            + dataset.describe_database_schema(db_name, mode="datagrip")
            + "\nColumn description:\n"
            + descriptions[col_name],
        },
    ]
    response = gcp_generator_candidate.generate(test_messages, temperature=TEMPERATURE)

    # Thread-safe update of the shared dictionary
    with llm_descriptions_lock:
        llm_descriptions[db_name][table_name][col_name] = response.text


def profile_database(db_path: str, column_meaning_path: str):
    gcp_generator_candidate = GCPGenerator(
        model=MODEL_NAME,
        api_key=GCP_KEY,
    )
    dataset = SqliteDataset(db_path)
    column_meaning_parsed = load_column_meaning(column_meaning_path)
    llm_descriptions = {}
    # Thread lock for safely updating the shared dictionary
    llm_descriptions_lock = threading.Lock()
    for db_name in dataset.get_databases():
        print(f"Processing {db_name}")
        if db_name not in llm_descriptions:
            llm_descriptions[db_name] = {}
        table_names = list(dataset.get_database_schema(db_name)["tables"].keys())
        for table_name in tqdm(table_names, desc=f"Processing tables in {db_name}"):
            if table_name == "sqlite_sequence":
                continue
            if table_name not in llm_descriptions[db_name]:
                llm_descriptions[db_name][table_name] = {}
            descriptions = profile_table(
                f"/Users/deni/Data/dev_databases/{db_name}/{db_name}.sqlite",
                table_name,
                column_meaning_parsed=column_meaning_parsed,
            )

            # Use ThreadPoolExecutor to process columns in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all column processing tasks
                futures = []
                for col_name in descriptions.keys():
                    future = executor.submit(
                        process_column,
                        db_name,
                        table_name,
                        col_name,
                        descriptions,
                        dataset,
                        gcp_generator_candidate,
                        llm_descriptions,
                        llm_descriptions_lock,
                    )
                    futures.append(future)

                # Wait for all tasks to complete with progress bar
                with tqdm(total=len(futures), desc=f"Processing columns in {table_name}") as pbar:
                    for future in as_completed(futures):
                        try:
                            future.result()  # This will raise any exceptions that occurred
                        except Exception as e:
                            print(f"Error processing column: {e}")
                        pbar.update(1)
    return llm_descriptions


def merge_descriptions(llm_descriptions: dict):
    new_descriptions = {}

    for llm_descriptions_db in llm_descriptions.keys():
        for llm_descriptions_table in llm_descriptions[llm_descriptions_db].keys():
            for llm_descriptions_col in llm_descriptions[llm_descriptions_db][llm_descriptions_table].keys():
                new_key = f"{llm_descriptions_db}|{llm_descriptions_table}|{llm_descriptions_col}"
                desc = llm_descriptions[llm_descriptions_db][llm_descriptions_table][llm_descriptions_col]
                new_descriptions[new_key] = desc
    return new_descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-database-path",
        type=str,
        required=True,
        help="path to the test databases base directory",
    )
    parser.add_argument(
        "--column-meaning-path",
        type=str,
        required=True,
        help="path to the column meaning file",
    )
    args = parser.parse_args()

    llm_descriptions = profile_database(args.test_database_path, args.column_meaning_path)

    with open("improved_column_meaning_parsed.json", "w") as f:
        json.dump(llm_descriptions, f, indent=2, ensure_ascii=False)
    # new_descriptions = merge_descriptions(llm_descriptions)
    # with open("improved_column_meaning.json", "w") as f:
    #     json.dump(new_descriptions, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
