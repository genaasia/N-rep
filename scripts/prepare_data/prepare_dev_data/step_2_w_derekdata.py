import copy
import json
import os
from pprint import pprint
from glob import glob
from copy import deepcopy
from text2sql.data.datasets import SqliteDataset
from tqdm import tqdm

from utils.mac_m_parser import parse_mac_schema, parse_m_schema
from text2sql.data.schema_to_text import (schema_to_basic_format,
                                          schema_to_datagrip_format,
                                          schema_to_sql_create)

def filter_schema_dict(schema_dict, filter_dict):
    schema_dict = deepcopy(schema_dict)
    pop_keys = []
    pop_cols = {}
    pop_fks = {}

    for table_name in schema_dict["tables"]:
        if table_name not in filter_dict:
            pop_keys.append(table_name)
            continue

        pop_cols[table_name] = []
        pop_fks[table_name] = []
        if filter_dict[table_name] == "keep_all":
            continue
        for col in schema_dict["tables"][table_name]["columns"]:
            if col not in filter_dict[table_name]:
                pop_cols[table_name].append(col)
        for fk_col in schema_dict["tables"][table_name]["foreign_keys"]:
            if fk_col not in filter_dict[table_name]:
                pop_fks[table_name].append(fk_col)
                continue
            referenced_table = schema_dict["tables"][table_name]["foreign_keys"][
                fk_col
            ]["referenced_table"]
            referenced_column = schema_dict["tables"][table_name]["foreign_keys"][
                fk_col
            ]["referenced_column"]
            if referenced_table not in filter_dict:
                pop_fks[table_name].append(fk_col)
                continue
            if referenced_column not in filter_dict[referenced_table]:
                pop_fks[table_name].append(fk_col)
                continue

    for key in pop_keys:
        schema_dict["tables"].pop(key)

    for table_name in pop_cols:
        for col in pop_cols[table_name]:
            schema_dict["tables"][table_name]["columns"].pop(col)

    for table_name in pop_fks:
        for col in pop_fks[table_name]:
            schema_dict["tables"][table_name]["foreign_keys"].pop(col)

    return schema_dict

def main():
    train_fn = "./mac_schema_hr3s_gemini_fulldev.json"
    schema_links_fn = "./dev_schema_links.json"
    DB_PATH = "/home/denizay/new_experiments/text2sql_testbed/scripts/run_pipeline/data/BIRD/dev_20240627/dev_databases"

    # Load the inference data
    with open(train_fn, "r") as f:
        inference_data = json.load(f)
    
    # Load the schema links data
    with open(schema_links_fn, "r") as f:
        schema_links_data = json.load(f)

    # Create a mapping from question_id to its schema links
    qid_to_schema_links = {}
    for item in schema_links_data:
        qid_to_schema_links[item["idx"]] = [candidate["mapping"] for candidate in item["candidates"]]

    train_db_schemadicts = {row["db_id"]: {} for row in inference_data}

    qid2parsed_generation_by_desc_type = {}
    for desc_type in ["mac_schema", "m_schema", "sql_create"]:
        if desc_type == "mac_schema":
            fn = "mac_schema_hr3s_gemini_fulldev.json"
        elif desc_type == "m_schema":
            fn = "m_schema_hr3s_fulldev.json"
        elif desc_type == "sql_create":
            fn = "sql_create_hr3s_fulldev.json"

        with open(fn, "r") as f:
            inference_data = json.load(f)
        qid2parsed_generation_by_desc_type[desc_type] = {row["question_id"]: row["parsed_generation"] for row in inference_data}

    sql_dataset = SqliteDataset(DB_PATH)
    for train_db_id in train_db_schemadicts:
        schema_dict = sql_dataset.get_database_schema(train_db_id)
        train_db_schemadicts[train_db_id] = schema_dict

    for desc_type in ["mac_schema", "m_schema", "sql_create"]:
        train_db_id_schemadescs = {row["db_id"]: "" for row in inference_data}
        for train_db_id in train_db_id_schemadescs:
            desc_file = os.path.join(f"../dev_descriptions/{desc_type}", f"{train_db_id}.txt")
            with open(desc_file, "r") as f:
                desc = f.read()
            train_db_id_schemadescs[train_db_id] = desc

        for idx, train_datum in enumerate(tqdm(inference_data)):
            try:
                question_id = train_datum["question_id"]
                if question_id not in qid_to_schema_links:
                    print(f"Warning: No schema links found for question_id {question_id}")
                    continue

                # Get all schema links for this question
                schema_links = qid_to_schema_links[question_id]
                
                # Process each schema linking
                for link_idx, schema_link in enumerate(schema_links):
                    if desc_type == "mac_schema":
                        filtered_schema_txt_table = parse_mac_schema(
                            train_db_id_schemadescs[train_datum["db_id"]],
                            schema_link,
                            force_keep_all=True,
                        )
                        filtered_schema_txt_col = parse_mac_schema(
                            train_db_id_schemadescs[train_datum["db_id"]],
                            schema_link,
                            force_keep_all=False,
                        )
                        inference_data[idx][f"mac_schema_table_filtered_schema_txt_{link_idx}"] = filtered_schema_txt_table
                        inference_data[idx][f"mac_schema_col_filtered_schema_txt_{link_idx}"] = filtered_schema_txt_col
                        inference_data[idx][f"mac_schema_full_schema_{link_idx}"] = train_db_id_schemadescs[train_datum["db_id"]]
                    elif desc_type == "m_schema":
                        filtered_schema_txt_table = parse_m_schema(
                            train_db_id_schemadescs[train_datum["db_id"]],
                            schema_link,
                            force_keep_all=True,
                        )
                        filtered_schema_txt_col = parse_m_schema(
                            train_db_id_schemadescs[train_datum["db_id"]],
                            schema_link,
                            force_keep_all=False,
                        )
                        inference_data[idx][f"m_schema_table_filtered_schema_txt_{link_idx}"] = filtered_schema_txt_table
                        inference_data[idx][f"m_schema_col_filtered_schema_txt_{link_idx}"] = filtered_schema_txt_col
                        inference_data[idx][f"m_schema_full_schema_{link_idx}"] = train_db_id_schemadescs[train_datum["db_id"]]
                    elif desc_type == "sql_create":
                        table_dict = copy.deepcopy(schema_link)
                        for key in table_dict.keys():
                            table_dict[key] = "keep_all"

                        schema_dict = train_db_schemadicts[train_datum["db_id"]]
                        filtered_schema_dict = filter_schema_dict(
                            schema_dict, schema_link
                        )
                        filtered_schema_dict_table = filter_schema_dict(
                            schema_dict, table_dict
                        )
                        
                        filtered_schema_txt_col = schema_to_sql_create(
                            train_datum["db_id"], filtered_schema_dict
                        )
                        filtered_schema_txt_table = schema_to_sql_create(
                            train_datum["db_id"], filtered_schema_dict_table
                        )
                        inference_data[idx][f"sql_create_table_filtered_schema_txt_{link_idx}"] = filtered_schema_txt_table
                        inference_data[idx][f"sql_create_col_filtered_schema_txt_{link_idx}"] = filtered_schema_txt_col
                        inference_data[idx][f"sql_create_full_schema_{link_idx}"] = train_db_id_schemadescs[train_datum["db_id"]]

            except Exception as e:
                print(f"Error processing question_id {train_datum.get('question_id', 'unknown')}: {str(e)}")
                raise e

        with open("dev_all_descs_multiple_links.json", "w") as f:
            json.dump(inference_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main() 