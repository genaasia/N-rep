import json
import os
import time
from copy import deepcopy
import multiprocessing
from functools import partial
from multiprocessing import Manager
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from sqlalchemy import create_engine
from text2sql.data.datasets import SqliteDataset
from text2sql.data.schema_to_text import (schema_to_basic_format,
                                          schema_to_datagrip_format,
                                          schema_to_sql_create)
from tqdm import tqdm

from description_writers.get_mschema import SchemaEngine
from description_writers.mac_m_parser import parse_m_schema, parse_mac_schema
from description_writers.minimal_mac_sql import Selector
from get_table_columns import get_table_mapping
from spacy_masker import extract_named_entities, replace_entities_with_tokens

TRAIN_DATABASES_PATH = "./train/train_databases"
TRAIN_DATA_PATH = "./train/train.json"
OUTPUT_MULTI_TABLE_PATH = "./outputs/valid_multi_table_queries.json"
OUTPUT_TRAIN_DATA_PATH = "./outputs/train_data_with_fields.json"
MAC_CACHE_PATH = "./train_mac_descriptions"

# We'll initialize these in the main function to avoid multiprocessing issues
schema_cache = None
mschema_cache = None

SUBSET = None


def load_data():
    """Load training data and initialize dataset."""
    sql_dataset = SqliteDataset(TRAIN_DATABASES_PATH)
    with open(TRAIN_DATA_PATH, "r") as f:
        train_data = json.load(f)
        if SUBSET:
            train_data = train_data[:SUBSET]
    return sql_dataset, train_data


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


def get_mschema_description(db_name):
    # Check cache first
    if db_name in mschema_cache:
        return mschema_cache[db_name]

    abs_path = os.path.join(TRAIN_DATABASES_PATH, db_name, f"{db_name}.sqlite")
    db_engine = create_engine(f"sqlite:///{abs_path}")
    schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
    mschema = schema_engine.mschema
    mschema_str = mschema.to_mschema()

    # Cache the result
    mschema_cache[db_name] = mschema_str
    return mschema_str


def get_cached_mac_description(db_id):
    """Read the pre-computed MAC description from the cached file."""
    desc_path = os.path.join(MAC_CACHE_PATH, f"{db_id}.txt")
    try:
        with open(desc_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: No cached description found for {db_id}")
        return None


def get_filtered_schema_txt(sql_dataset, table_map, db_id, selector):
    # Get schema from cache or database
    if db_id in schema_cache:
        schema_dict = schema_cache[db_id]
    else:
        schema_dict = sql_dataset.get_database_schema(db_id)
        schema_cache[db_id] = schema_dict

    filtered_schema_dict = filter_schema_dict(schema_dict, table_map)
    dail_types_relations_text = schema_to_basic_format(
        db_id,
        filtered_schema_dict,
        include_types=True,
        include_relations=True,
    )
    sql_create_text = schema_to_sql_create(db_id, filtered_schema_dict)
    json_raw_text = schema_to_datagrip_format(db_id, filtered_schema_dict)

    m_full_text = get_mschema_description(db_id)
    m_schema_text = parse_m_schema(m_full_text, table_map, force_keep_all=False)

    # Use cached MAC description instead of generating it
    mac_full_text = get_cached_mac_description(db_id)
    if mac_full_text is None:
        # Fallback to original method if cache miss
        mac_full_text = selector._get_db_desc_str(db_id)[0]
    mac_schema_text = parse_mac_schema(mac_full_text, table_map, force_keep_all=False)

    return {
        "dail_types_relations": dail_types_relations_text,
        "sql_create": sql_create_text,
        "json_raw": json_raw_text,
        "m_schema": m_schema_text,
        "mac_schema": mac_schema_text,
    }


def process_single_query(item, sql_dataset, selector):
    """Process a single query and return its validation status."""
    # Create a copy to avoid modifying the original
    item = item.copy()  
    
    db_id = item["db_id"]
    sql_query = item["sql_query"]
    t0 = time.time()
    
    try:
        # Get schema info for this database
        schema_time = 0
        t0_schema = time.time()
        schema_info = sql_dataset.get_database_schema(db_id)
        schema_time = time.time() - t0_schema

        # Get table mapping for this query
        mapping_time = 0
        t1_mapping = time.time()
        table_mapping = get_table_mapping(schema_info, sql_query)
        mapping_time = time.time() - t1_mapping

        # Get filtered schema
        schema_filter_time = 0
        t2_filter = time.time()
        filtered_schema_dict = get_filtered_schema_txt(
            sql_dataset, table_mapping["table_map"], db_id, selector
        )
        schema_filter_time = time.time() - t2_filter

        item.update(filtered_schema_dict)

        # Update table count in train data
        item["table_count"] = len(table_mapping["tables"])

        # If query uses more than one table, validate and execute it
        if len(table_mapping["tables"]) > 1:
            validation_time = 0
            t3_validation = time.time()
            result = sql_dataset.validate_query(db_id, sql_query, timeout_secs=5)
            validation_time = time.time() - t3_validation

            if result["validated"] or result["message"].startswith("query timed out"):
                ner_time = 0
                t4_ner = time.time()
                entities = extract_named_entities(item["nl_en_query"])
                item["nl_en_query_masked"] = replace_entities_with_tokens(item["nl_en_query"], entities)
                ner_time = time.time() - t4_ner
                
                # Add timing information to the item
                item["timing"] = {
                    "schema_lookup": schema_time,
                    "table_mapping": mapping_time,
                    "schema_filtering": schema_filter_time,
                    "query_validation": validation_time,
                    "ner_extraction": ner_time,
                    "total": time.time() - t0
                }
                return item, True, False, False
            else:
                # Add timing information even for invalid queries
                item["timing"] = {
                    "schema_lookup": schema_time,
                    "table_mapping": mapping_time,
                    "schema_filtering": schema_filter_time,
                    "query_validation": validation_time,
                    "ner_extraction": 0,
                    "total": time.time() - t0
                }
                return item, False, True, False
        else:
            # Add timing information for single-table queries
            item["timing"] = {
                "schema_lookup": schema_time,
                "table_mapping": mapping_time,
                "schema_filtering": schema_filter_time,
                "query_validation": 0,
                "ner_extraction": 0,
                "total": time.time() - t0
            }
            return item, False, False, True

    except Exception as e:
        print(f"Error processing query for db {db_id}: {str(e)}")
        # Add timing information even for failed queries
        schema_time = schema_time if 'schema_time' in locals() else 0
        mapping_time = mapping_time if 'mapping_time' in locals() else 0
        schema_filter_time = schema_filter_time if 'schema_filter_time' in locals() else 0
        
        item["timing"] = {
            "schema_lookup": schema_time,
            "table_mapping": mapping_time,
            "schema_filtering": schema_filter_time,
            "query_validation": 0,
            "ner_extraction": 0,
            "total": time.time() - t0
        }
        return item, False, True, False


def save_results(valid_multi_table_queries, train_data):
    """Save the processed results to files."""
    for i, item in enumerate(valid_multi_table_queries):
        if "timing" in valid_multi_table_queries[i]:
            del valid_multi_table_queries[i]["timing"]

    with open(OUTPUT_MULTI_TABLE_PATH, "w") as f:
        json.dump(valid_multi_table_queries, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_TRAIN_DATA_PATH, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)


def print_statistics(
    train_data, valid_multi_table_queries, single_table_count, invalid_query_count
):
    """Print processing statistics."""
    print(f"Original queries: {len(train_data)}")
    print(f"Valid multi-table queries: {len(valid_multi_table_queries)}")
    print(f"Single-table queries removed: {single_table_count}")
    print(f"Invalid queries removed: {invalid_query_count}")
    print(f"Total queries removed: {len(train_data) - len(valid_multi_table_queries)}")


def worker_init(train_databases_path, selector_params):
    """Initialize worker process with necessary resources."""
    global sql_dataset, worker_selector
    
    # Initialize dataset for this worker
    sql_dataset = SqliteDataset(train_databases_path)
    
    # Initialize selector for this worker
    worker_selector = Selector(
        train_databases_path,
        tables_json_path=selector_params['tables_json_path'],
        dataset_name=selector_params['dataset_name'],
        lazy=selector_params['lazy']
    )


def process_query_wrapper(item, worker_id):
    """Wrapper function for the worker processes."""
    global sql_dataset, worker_selector
    return process_single_query(item, sql_dataset, worker_selector)


def process_chunk(chunk, sql_dataset, selector):
    """Process a chunk of queries in a single process."""
    results = []
    for item in chunk:
        result = process_single_query(item, sql_dataset, selector)
        results.append(result)
    return results


def process_queries():
    """Main function to process and filter queries using ProcessPoolExecutor."""
    # Initialize and load data
    global schema_cache, mschema_cache
    
    t_start = time.time()
    sql_dataset, train_data = load_data()
    
    # Initialize shared caches
    manager = Manager()
    schema_cache = manager.dict()
    mschema_cache = manager.dict()
    
    selector = Selector(
        TRAIN_DATABASES_PATH,
        tables_json_path="./description_writers/train_tables.json",
        dataset_name="bird",
        lazy=True,
    )
    
    init_time = time.time() - t_start
    print(f"Initialization time: {init_time:.2f} seconds")

    # Determine number of processes to use (limit to avoid overloading)
    num_processes = min(multiprocessing.cpu_count(), 8)
    print(f"Using {num_processes} processes for parallel processing")

    # Divide data into chunks for each process
    chunk_size = (len(train_data) + num_processes - 1) // num_processes
    chunks = [train_data[i:i + chunk_size] for i in range(0, len(train_data), chunk_size)]
    
    # Create a progress bar that will work with multiprocessing
    total_items = len(train_data)
    progress_bar = tqdm(total=total_items, desc="Processing queries")
    
    # Process each chunk in its own process
    all_results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(process_chunk, chunk, sql_dataset, selector): len(chunk) 
                          for chunk in chunks}
        
        # As each task completes, update the progress bar
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_size = future_to_chunk[future]
            try:
                results = future.result()
                all_results.extend(results)
                # Update progress bar by the chunk size
                progress_bar.update(chunk_size)
            except Exception as exc:
                print(f'A chunk generated an exception: {exc}')
                # Still update the progress bar even if there was an error
                progress_bar.update(chunk_size)
    
    # Close the progress bar
    progress_bar.close()

    # Process results
    valid_multi_table_queries = []
    single_table_count = 0
    invalid_query_count = 0
    total_timing = {
        "schema_lookup": 0,
        "table_mapping": 0,
        "schema_filtering": 0,
        "query_validation": 0,
        "ner_extraction": 0,
        "total": 0
    }

    for processed_item, is_valid, is_invalid, is_single_table in all_results:
        # Accumulate timing information
        for key in total_timing:
            total_timing[key] += processed_item["timing"][key]

        if is_valid:
            valid_multi_table_queries.append(processed_item)
        elif is_invalid:
            invalid_query_count += 1
        elif is_single_table:
            single_table_count += 1

    # Print timing statistics
    total_time = time.time() - t_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    print("\nTiming Statistics (averages per query):")
    print("Total queries processed:", len(train_data))
    for key, value in total_timing.items():
        avg_time = value / len(train_data)
        print(f"{key}: {avg_time:.3f} seconds")

    # Save results and print statistics
    save_results(valid_multi_table_queries, train_data)
    print_statistics(
        train_data, valid_multi_table_queries, single_table_count, invalid_query_count
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows compatibility
    process_queries()