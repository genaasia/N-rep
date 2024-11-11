import os
import time
import json
import argparse
import sys
from time import time, sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
from tqdm.auto import tqdm
from requests.exceptions import RequestException

# import package relatively
try:
    import text2sql
except:
    print("text2sql not installed, adding src to path")
    sys.path.append("../../src")

from plotter import plot_accuracy
from text2sql.evaluation.metrics import (
    df_sql_match,
    df_execution_match,
    df_intent_match,
    df_soft_f1_score,
    df_template_match,
)

from settings import get_settings_vars


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", default="config.yaml", help="Path to config file")
parser.add_argument("--subset", "-n", type=int, default=-1, help="Dataset subset size to use for testing, -1 == all")
ARGS = parser.parse_args()

(
    URL,
    GEN_MODELS,
    GEN_MODS,
    NUM_OF_WORKERS,
    FILE_NAME,
    OUT_FOLDER,
    PLOT_FOLDER,
    RESULTS_TAG,
    RESULTS_FILE_NAME,
    NLQ_COL,
    EX_LABEL_COL,
    SQL_COL,
    TEMPLATE_ID_COL,
    PRED_EX_LABEL_COL,
    PRED_SQL_COL,
    PRED_TEMPLATE_ID_COL,
    METRICS,
    PLOT_LABELS,
    EVAL_DATA_CONFIG,
), settings = get_settings_vars(ARGS.config)


metric_name2func = {
    "execution_match": df_execution_match,
    "sql_match": df_sql_match,
    "intent_match": df_intent_match,
    "soft_f1_score": df_soft_f1_score,
    "template_match": df_template_match,
}

session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))


def make_request_with_retry(session, url, payload, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            response = session.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
            )
            response.raise_for_status()  # Raises an exception for 4XX/5XX status codes
            return response.json()

        except RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                # debug only
                print(json.dumps(payload, indent=2))
                raise  # Re-raise the last exception if all retries failed
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            sleep(delay)


with open("base_payload.json", "rb") as f:
    BASE_PAYLOAD = json.load(f)


def run_text2sql(nl_query, gen_backend, gen_model, gen_mode):
    # if not (isinstance(user_id, str) and user_id.isnumeric()):
    #     user_id = dummy_id

    payload = BASE_PAYLOAD.copy()
    payload.update(
        {
            "query": nl_query,
            # "template_kwargs": {"client_user_id": user_id},
            "gen_mode": gen_mode,
            "gen_backend": gen_backend,
            "gen_model": gen_model,
            # "language": settings.language,
            "client": settings.client,
            "dataset": settings.dataset,
            "db_type": settings.db_type,
            "db_name": settings.db_name,
            "collection": settings.collection,
        }
    )

    result = make_request_with_retry(session, URL, payload)

    failed = False
    database_fail = False

    if result.get("generation_response"):
        sql_pred = result["generation_response"].get("response")
    else:
        failed = True
        sql_pred = None
    if result.get("database_response"):
        ex_pred = result["database_response"].get("response")
        if result["database_response"]["status"] == "error":
            database_fail = True
    else:
        failed = True
        ex_pred = None
    # allow zero-shot without template match
    try:
        template_first_pred = result["retrieval_response"]["results"][0]["properties"]["info_id"]
    except IndexError:
        template_first_pred = None
    if template_first_pred is None:
        named_metrics = [m["name"] for m in METRICS]
        if "template_match" in named_metrics:
            raise ValueError(f"Template match metric is enabled but no template was found for query: {nl_query}")
    return ex_pred, sql_pred, template_first_pred, failed, database_fail


# def get_dummy_user_id(df, user_id_col):
#     dummy_id = next((x for x in df[user_id_col] if x.isnumeric()), None)
#     return dummy_id


def process_single_row(args):
    """Process a single row with the given arguments"""
    row, nlq_col, gen_backend, gen_model, gen_mode = args
    ex_pred, sql_pred, template_pred, failed, database_fail = run_text2sql(
        row[nlq_col], gen_backend, gen_model, gen_mode
    )
    return {
        "index": row.name,
        PRED_EX_LABEL_COL: ex_pred,
        PRED_SQL_COL: sql_pred,
        PRED_TEMPLATE_ID_COL: template_pred,
        "Fail": failed,
        "Database Error": database_fail,
    }


def parallel_run_text2sql(df, gen_backend, gen_model, gen_mode, subset: int = -1, max_workers=NUM_OF_WORKERS):
    """
    Run text2sql in true parallel using ThreadPoolExecutor

    Args:
        df: Input dataframe
        gen_backend: Generation backend to use
        gen_model: Generation backend, model to use
        gen_mode: Generation mode to use
        subset: only do subset of data (-1: all)
        max_workers: Number of parallel threads to use
    """
    # for debugging
    if subset > 0:
        print(f"Running on subset of {subset} rows")
        df = df.truncate(after=subset - 1)

    # Prepare arguments for each row
    row_args = [(row, NLQ_COL, gen_backend, gen_model, gen_mode) for _, row in df.iterrows()]

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(process_single_row, args): args for args in row_args}

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_row), total=len(row_args)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing row: {type(e).__name__}: {str(e)}")
                continue

    # Convert results to DataFrame and sort by original index
    # print(results)
    results_df = pd.DataFrame(results)
    results_df.set_index("index", inplace=True)
    results_df.sort_index(inplace=True)

    # Update the original dataframe with results
    df[PRED_EX_LABEL_COL] = results_df[PRED_EX_LABEL_COL]
    df[PRED_SQL_COL] = results_df[PRED_SQL_COL]
    df[PRED_TEMPLATE_ID_COL] = results_df[PRED_TEMPLATE_ID_COL]
    df["Fail"] = results_df["Fail"]
    df["Database Error"] = results_df["Database Error"]

    # Parse the execution labels
    try:
        df[EX_LABEL_COL] = df[EX_LABEL_COL].apply(json.loads)
    except Exception as e:
        # print(f"df['{EX_LABEL_COL}']: {df[EX_LABEL_COL]}")
        raise
    return df


def get_accuracy(gen_backend, gen_model, gen_mode):
    usecols = [NLQ_COL, EX_LABEL_COL, SQL_COL, TEMPLATE_ID_COL]
    # df = pd.read_csv(FILE_NAME, usecols=usecols, dtype={USER_ID_COL: str}, nrows=10)
    df = pd.read_csv(FILE_NAME, usecols=usecols)

    # dummy_id = get_dummy_user_id(df)

    # if not dummy_id:
    #     raise Exception("Couldn't find any dummy user id.")

    eval_results = {}

    for col in [PRED_EX_LABEL_COL, PRED_SQL_COL, PRED_TEMPLATE_ID_COL]:
        df[col] = None

    start = time()
    df = parallel_run_text2sql(df, gen_backend, gen_model, gen_mode, ARGS.subset, max_workers=8)
    eval_results["Failed Ratio"] = (df["Fail"].sum() / len(df.index)) * 100
    eval_results["Database Error Ratio"] = (df["Database Error"].sum() / len(df.index)) * 100
    print(f"Took {time()-start} seconds.")

    for metric_config in METRICS:
        # print(f"Calculating {metric_config["name"]}")
        metric = metric_name2func[metric_config["name"]]
        df = metric(df, eval_results, EVAL_DATA_CONFIG, metric_config["output_column"])

    out_file_name = os.path.basename(FILE_NAME)
    out_file_name = os.path.splitext(out_file_name)[0]
    out_file_name = "__".join([out_file_name, gen_model, gen_mode, RESULTS_TAG])
    out_file_name = out_file_name + ".csv"
    out_file_name = os.path.join(OUT_FOLDER, out_file_name)

    columns_ordered = [
        NLQ_COL,
        TEMPLATE_ID_COL,
        PRED_TEMPLATE_ID_COL,
        SQL_COL,
        PRED_SQL_COL,
        EX_LABEL_COL,
        PRED_EX_LABEL_COL,
    ]

    for col in df.columns:
        if col not in columns_ordered:
            columns_ordered.append(col)

    df = df[columns_ordered]

    if not os.path.isdir(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    df.to_csv(out_file_name, index=False)

    return eval_results


def get_accuracies(gen_model_configs, gen_modes):
    all_results = {}

    for gen_model in gen_model_configs:
        gen_backend = gen_model.get("backend")
        gen_model = gen_model.get("model")
        evaluated_models = []
        accuracies = {}
        plot_labels = PLOT_LABELS.copy()

        for gen_mode in gen_modes:
            print(f"Getting accuracies for gen model: {gen_backend} {gen_model} and gen mode: {gen_mode}")

            model_results = get_accuracy(gen_backend, gen_model, gen_mode)
            evaluated_models.append(f"{gen_model}\n{gen_mode}")

            for acc_type, acc_results in model_results.items():
                if not isinstance(acc_results, dict):
                    pass
                else:
                    if acc_type not in accuracies:
                        accuracies[acc_type] = []
                    accuracies[acc_type].append(acc_results["accuracy"])

            if "Database Error Ratio" in model_results:
                if "Database Error Ratio" not in accuracies:
                    plot_labels.append("Database Error Ratio")
                    accuracies["Database Error Ratio"] = []
                accuracies["Database Error Ratio"].append(model_results["Database Error Ratio"])

            all_results[f"{gen_model}__{gen_mode}"] = model_results

            with open(RESULTS_FILE_NAME, "w") as f:
                json.dump(all_results, f, indent=2)

        plot_file_name = os.path.join(
            PLOT_FOLDER,
            os.path.basename(FILE_NAME).replace(".csv", f"_{gen_model}({RESULTS_TAG}).png"),
        )
        if not os.path.isdir(PLOT_FOLDER):
            os.makedirs(PLOT_FOLDER)
        plot_accuracy(
            evaluated_models,
            accuracies.values(),
            plot_labels,
            plot_file_name,
        )


def main():
    get_accuracies(GEN_MODELS, GEN_MODS)


if __name__ == "__main__":
    main()
