import re


# Pattern to match Table definitions
def parse_table(text: str):
    """parse detected `Table` into table and columns"""
    table_parse_pattern = (
        r"t_\w+\s*=\s*Table\(\s*'([^']+)',\s*\w+,\s*((?:.*?\),\s*)*(?:.*?\)))"
    )
    matches = re.findall(table_parse_pattern, text, re.DOTALL)
    tables = []
    for table_name, columns_text in matches:
        # Process columns if needed
        columns = re.findall(r"Column\('([^']+)'", columns_text)
        tables.append({"table": table_name, "columns": columns})
    return tables[0]


# regex that captures the `Table` objects starting with e.g. "t_schools = Table(" and ending with empty line
def extract_table_blocks(code):
    """detect `Table` blocks in the code"""
    table_pattern = r"(t_\w+\s*=\s*Table\(\s*'[^']+',\s*metadata,[\s\S]*?\n\))"
    table_blocks = re.findall(table_pattern, code, re.DOTALL)
    return table_blocks


def process_one_sqlalchemy_schema(schema_text: str, filtered_schema: dict) -> str:
    # get header with imports and metadata definition
    header = schema_text.split("\n\n\n")[0] + "\n\n"
    # extract tables
    tables = extract_table_blocks(schema_text)
    truncated_tables = []
    for table in tables:
        table_info = parse_table(table)
        table_name = table_info["table"]
        # only process ones in schema selector prediction
        if table_name in list(filtered_schema.keys()):
            lines = table.split("\n")
            keep_columns = filtered_schema[table_name]
            truncated_lines = []
            for line in lines:
                # remove Column() lines if column name not in selected columns
                if "Column(" in line:
                    column_name_match = re.search(r"Column\('([^']+)'", line)
                    if column_name_match:
                        column_name = column_name_match.group(1)
                        if keep_columns == "keep_all" or column_name in keep_columns:
                            # if info like "ForeignKey('foreign_table.column')," is present,
                            # remove it if the foreign_table or column is not in the filtered schema
                            if "ForeignKey(" in line:
                                foreign_key_match = re.search(
                                    r"ForeignKey\('([^']+)\.([^']+)'\),\s", line
                                )
                                if foreign_key_match:
                                    foreign_table, foreign_column = (
                                        foreign_key_match.groups()
                                    )
                                    if (
                                        foreign_table not in filtered_schema.keys()
                                        or foreign_column
                                        not in filtered_schema[foreign_table]
                                    ):
                                        # Just remove the ForeignKey part but keep the rest of the `Column`
                                        line = line.replace(
                                            foreign_key_match.group(), ""
                                        )
                            truncated_lines.append(line)
                else:
                    truncated_lines.append(line)

            truncated_tables.append("\n".join(truncated_lines) + "\n\n")

    return "\n".join([header] + truncated_tables).strip("\n") + "\n"


if __name__ == "__main__":
    import argparse
    import glob
    import json
    import os

    # read parameter "predictio_-file", "base_directory" and "file extension" from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-file",
        type=str,
        required=True,
        help="path to prediction json file",
    )
    parser.add_argument(
        "--base-directory",
        type=str,
        required=True,
        help="path to sqlalchemy schema text file directory",
    ),
    parser.add_argument(
        "--file-extension",
        type=str,
        default="txt",
        help="file extension of the schema text files",
    )
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join({args.base_directory}, f"*.{args.file_extension}")),
        key=lambda x: os.path.basename(x),
    )
    print(f"Found {len(files)} target files:")
    for file in files:
        print(file)
    print()

    file_mapping = dict()
    for file in files:
        base_name = os.path.basename(file).split(".")[0]
        file_mapping[base_name] = file

    with open(args.prediction_file) as f:
        predicted_schema = json.load(f)
    print(f"loaded {len(predicted_schema)} predicted samples")
    print()

    results = []
    for prediction in predicted_schema:
        db_id = prediction["db_id"]
        fpath = file_mapping[db_id]
        with open(fpath) as f:
            schema_text = f.read()
        filtered_schema = prediction["parsed_generation"]
        filtered_text = process_one_sqlalchemy_schema(schema_text, filtered_schema)
        results.append(
            {"filtered_schema": filtered_schema, "filtered_text": filtered_text}
        )
    print(f"successfully processed {len(results)} schema files")
    print()
    print(f"printing some examples:")

    for idx in range(0, 100, 10):
        print(
            f"\n=============================================\nquestion idx {idx}\npredicted tables and columns:\n"
        )
        filtered_schema = results[idx]["filtered_schema"]
        filtered_text = results[idx]["filtered_text"]
        print(json.dumps(filtered_schema, indent=2))
        print(
            "\n=============================================\nprocessed schema text:\n\n"
        )
        print(filtered_text)
        print("\n")
