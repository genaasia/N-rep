import argparse
import json
import os

import pandas as pd

from rapidfuzz import fuzz


# args = Args()
parser = argparse.ArgumentParser("convert results to tab-separated eval format")
parser.add_argument('--source-file', type=str, required=True, help='path to gena output file')
parser.add_argument('--target-path', type=str, required=True, help='path to directory of saved results')
args = parser.parse_args()


# load the source file
with open(args.source_file, 'r') as f:
    source_data = json.load(f)
# load target file "predict_dev.json"
with open(os.path.join(args.target_path, 'predict_dev.json'), 'r') as f:
    target_data = json.load(f)
# load execution results "exec_result.json"
with open(os.path.join(args.target_path, 'exec_result.json'), 'r') as f:
    exec_result = json.load(f)
assert len(source_data) == len(target_data) == len(exec_result)


def proc_result(idx: int):
    row = target_data[str(idx)]
    return row.split("\t")[0]


df_rows: list[dict] = []
for idx in range(len(source_data)):
    row = {
        "question_id": source_data[idx]['question_id'],
        "db_id": source_data[idx]['db_id'],
        "original_question": source_data[idx]['source_question'],
        "question": source_data[idx]['question'],
        "evidence": source_data[idx]['evidence'],
        "gold_sql": source_data[idx]['SQL'],
        "predicted_sql": proc_result(idx),
        "question_token_ratio": fuzz.token_sort_ratio(source_data[idx]['question'], source_data[idx]['source_question']),
        "sql_query_token_ratio": fuzz.token_sort_ratio(source_data[idx]['SQL'], proc_result(idx)),
        "ex_accuracy": exec_result[idx]["res"]
    }
    df_rows.append(row)


    # save output dataframe to dir
with open(os.path.join(args.target_path, 'dataframe.csv'), 'w') as f:
    df = pd.DataFrame(df_rows)
    df.to_csv(f, index=False)

print("done!")