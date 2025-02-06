from ast import literal_eval

import numpy as np
from .eval_utils import (
    upper_bound_eval,
    lower_bound_eval,
    highest_voted_eval,
    highest_voted_valid_eval,
)


def run_eval(predicted_data, target_data, score_cache, target_sql_key):
    all_methods = {}
    # for method in ["highest_voted_valid", "highest_voted", "upper_bound", "lower_bound"]:
    for method in ["highest_voted_valid"]:
        all_methods[method] = {
            "sql_match_scores": [],
            "execution_match_scores": [],
            "intent_scores": [],
            "soft_f1_scores": [],
            "valids": [],
        }
        for i, predicted_datum in enumerate(predicted_data):
            target_datum = target_data[i]
            predictions = predicted_datum["predictions"]
            if not predictions:
                all_methods[method]["sql_match_scores"].append(False)
                all_methods[method]["execution_match_scores"].append(False)
                all_methods[method]["intent_scores"].append(False)
                all_methods[method]["soft_f1_scores"].append(0.0)
                all_methods[method]["valids"].append(False)
                continue

            target_sql = target_datum[target_sql_key]

            if isinstance(target_datum["api_execution_result"], str):
                target_execution = literal_eval(target_datum["api_execution_result"])
            else:
                target_execution = target_datum["api_execution_result"]

            if method == "upper_bound":
                (
                    sql_match_score,
                    execution_match_score,
                    intent_score,
                    soft_f1_score,
                    has_valid,
                ) = upper_bound_eval(
                    predictions, target_sql, target_execution, score_cache
                )
            elif method == "lower_bound":
                (
                    sql_match_score,
                    execution_match_score,
                    intent_score,
                    soft_f1_score,
                    has_valid,
                ) = lower_bound_eval(
                    predictions, target_sql, target_execution, score_cache
                )
            elif method == "highest_voted":
                (
                    sql_match_score,
                    execution_match_score,
                    intent_score,
                    soft_f1_score,
                    has_valid,
                ) = highest_voted_eval(
                    predictions, target_sql, target_execution, score_cache
                )

            elif method == "highest_voted_valid":
                (
                    sql_match_score,
                    execution_match_score,
                    intent_score,
                    soft_f1_score,
                    has_valid,
                    predicted_sql,
                    predicted_execution,
                ) = highest_voted_valid_eval(
                    predictions, target_sql, target_execution, score_cache
                )
                predicted_data[i]["highest_voted_valid"] = {
                    "intent_score": intent_score,
                    "predicted_sql": predicted_sql,
                    "predicted_execution": predicted_execution,
                }
            else:
                raise Exception(f"Method {method} is not supported")

            all_methods[method]["sql_match_scores"].append(sql_match_score)
            all_methods[method]["execution_match_scores"].append(execution_match_score)
            all_methods[method]["intent_scores"].append(intent_score)
            all_methods[method]["soft_f1_scores"].append(soft_f1_score)
            all_methods[method]["valids"].append(has_valid)

    all_methods_meaned = {}
    for key, values in all_methods.items():
        if not key in all_methods_meaned:
            all_methods_meaned[key] = {}
        for eval_key, eval_values in values.items():
            all_methods_meaned[key][eval_key] = np.mean(eval_values) * 100

    return all_methods_meaned
