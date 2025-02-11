import argparse
import json
import os
import re
import sqlite3
import time

from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import tqdm

from dbfunctions import query_sqlite_database

from metrics import (
    get_execution_match,
    get_soft_f1_score,
    get_intent_match,
)


def evaluate_one_sample(row) -> dict:
    database = row["db_id"]
    gold_sql = row["SQL"]
    predicted_sql = row["predicted_SQL"]

    # query database
    gold_status, gold_execution = query_sqlite_database(args.database_directory, database, gold_sql)
    pred_status, pred_execution = query_sqlite_database(args.database_directory, database, predicted_sql)

    # calculate metrics
    soft_f1 = get_soft_f1_score(pred_execution, gold_execution)
    intent_match = get_intent_match(pred_execution, gold_execution)
    execution_match = get_execution_match(pred_execution, gold_execution)

    return {
        "status_gold": gold_status,
        "status_pred": pred_status,
        "soft_f1": soft_f1,
        "intent_match": intent_match,
        "execution_match": execution_match,
    }


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="inference data")
    parser.add_argument("--prediction-file", type=str, required=True, help="path to prediction json file")
    parser.add_argument("--database-directory", type=str, required=True, help="path base directory of databases")
    parser.add_argument("--threads", type=int, default=4, help="number of threads (parallel jobs) to use. default: 4")
    args = parser.parse_args()

    if not os.path.exists(args.prediction_file):
        raise FileNotFoundError(f"prediction file not found: {args.prediction_file}")
    if not os.path.isdir(args.database_directory):
        raise NotADirectoryError(f"database directory not found: {args.database_directory}")

    # load dev data
    with open(args.prediction_file, "r") as f:
        dev_data: list[dict] = json.load(f)
    print(f"Loaded {len(dev_data)} examples from {args.prediction_file}")

    # run evaluation
    print(f"Running evaluation on {len(dev_data)} samples")
    time.sleep(1)
    with ThreadPool(args.threads) as pool:
        prediction_output_list: list[str] = list(tqdm.tqdm(pool.imap(evaluate_one_sample, dev_data), total=len(dev_data)))
    print(f"Finished inference on {len(dev_data)} examples")

    # calculate averages and display
    print("Evaluation summary:")
    soft_f1_scores = [r["soft_f1"] for r in prediction_output_list]
    intent_match_scores = [int(r["intent_match"]) for r in prediction_output_list]
    execution_match_scores = [int(r["execution_match"]) for r in prediction_output_list]
    print(f"Soft F1        : {np.mean(soft_f1_scores):.3f}")
    print(f"Intent Match   : {np.mean(intent_match_scores):.3f}")
    print(f"Execution Match: {np.mean(execution_match_scores):.3f}")

    # merge into data and save results as csv
    for i, row in enumerate(dev_data):
        row.update(prediction_output_list[i])
    df = pd.DataFrame(dev_data)
    output_file = f"{args.prediction_file.replace('.json', '')}_evaluation.csv"
    df.to_csv(output_file, index=False)