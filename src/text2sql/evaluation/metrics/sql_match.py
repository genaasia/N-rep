from multiprocessing import Pool, cpu_count
from txt2sql.metrics import sql_match
import pandas as pd
from time import time

from .match_utils import get_match_for_valid_exec


def process_chunk(args):
    chunk_index, chunk, sql_col, pred_sql_col = args
    result = chunk.apply(
        lambda row: get_match_for_valid_exec("sql_match", sql_match, row[pred_sql_col], row[sql_col], row["Fail"]),
        axis=1,
    )
    return chunk_index, result


def df_sql_match(df, results, config, out_col, n_processes=8):
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    # Split dataframe into chunks for parallel processing
    chunk_size = len(df) // n_processes
    chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Create process arguments with chunk indices
    process_args = [(i, chunk, config.sql_col, config.pred_sql_col) for i, chunk in enumerate(chunks)]

    start = time()
    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        results_with_indices = pool.map(process_chunk, process_args)
    end = time()
    print(f"Took {end-start} to normalize sqls")
    # Sort results by chunk index and combine
    sorted_results = sorted(results_with_indices, key=lambda x: x[0])
    df[out_col] = pd.concat([result for _, result in sorted_results])

    # Calculate metrics
    true_count = df[out_col].sum().item()
    false_count = len(df) - true_count
    accuracy = (true_count / (true_count + false_count)) * 100

    # Store results
    results["sql_match"] = {
        "matched": true_count,
        "different": false_count,
        "accuracy": accuracy,
    }

    return df
