import json


def create_db_table_column_mapping(json_file_path):
    """
    Read a JSON file containing database schema information and create a mapping
    of databases to their tables with column counts.

    Args:
        json_file_path (str): Path to the JSON file

    Returns:
        dict: A nested dictionary with the structure {db_id: {table_name: column_count}}
    """
    # Read the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Create the mapping
    db_table_col_count = {}

    for db in data:
        db_id = db["db_id"]
        db_table_col_count[db_id] = {}

        # Get the table names
        table_names = db["table_names_original"]

        # Get column names and their associated table indices
        # Skip the first entry which is [-1, "*"]
        column_names = db["column_names"][1:]

        # Count columns per table
        for col in column_names:
            table_idx = col[0]  # The table index
            if table_idx >= 0 and table_idx < len(table_names):
                table_name = table_names[table_idx]
                if table_name in db_table_col_count[db_id]:
                    db_table_col_count[db_id][table_name] += 1
                else:
                    db_table_col_count[db_id][table_name] = 1

    return db_table_col_count


# Example usage
if __name__ == "__main__":
    result = create_db_table_column_mapping("tables.json")

    # Print sample of the mapping
    print("Sample of the mapping:")
    count = 0
    for db_id, tables in result.items():
        print(f"\nDatabase: {db_id}")
        print(tables)
        count += 1
        if count >= 3:
            break

    print(f"\nTotal number of databases: {len(result)}")

    # Optionally save the result to a new JSON file
    with open("db_table_column_counts.json", "w") as f:
        json.dump(result, f, indent=2)
