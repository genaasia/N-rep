from ast import literal_eval
import tqdm
import numpy as np
from .eval_utils import (
    upper_bound_eval,
    lower_bound_eval,
    highest_voted_eval,
    highest_voted_valid_eval,
)


def run_eval(predicted_data, target_data, score_cache, target_sql_key, metrics=None):
    if metrics is None:
        metrics = ["sql_match", "execution_match", "intent", "soft_f1"]  # Default to all metrics if not specified
    
    all_methods = {}
    # for method in ["highest_voted_valid", "highest_voted", "upper_bound", "lower_bound"]:
    for method in ["highest_voted_valid"]:
        all_methods[method] = {
            "valids": [],
        }
        # Initialize metric-specific lists based on requested metrics
        for metric in metrics:
            all_methods[method][f"{metric}_scores"] = []
            
        for i, predicted_datum in enumerate(tqdm.tqdm(predicted_data)):
            target_datum = target_data[i]
            predictions = predicted_datum["predictions"]
            if not predictions:
                for metric in metrics:
                    all_methods[method][f"{metric}_scores"].append(False)
                all_methods[method]["valids"].append(False)
                continue

            target_sql = target_datum[target_sql_key]

            if isinstance(target_datum["api_execution_result"], str):
                target_execution = literal_eval(target_datum["api_execution_result"])
            else:
                target_execution = target_datum["api_execution_result"]

            if method == "upper_bound":
                scores = upper_bound_eval(
                    predictions, target_sql, target_execution, score_cache, metrics
                )
                has_valid = scores[-1]
                metric_scores = scores[:-1]
            elif method == "lower_bound":
                scores = lower_bound_eval(
                    predictions, target_sql, target_execution, score_cache, metrics
                )
                has_valid = scores[-1]
                metric_scores = scores[:-1]
            elif method == "highest_voted":
                scores = highest_voted_eval(
                    predictions, target_sql, target_execution, score_cache, metrics
                )
                has_valid = scores[-1]
                metric_scores = scores[:-1]
            elif method == "highest_voted_valid":
                scores = highest_voted_valid_eval(
                    predictions, target_sql, target_execution, score_cache, metrics
                )
                has_valid = scores[-3]
                metric_scores = scores[:-3]
                predicted_sql = scores[-2]
                predicted_execution = scores[-1]
                
                # Store all metric scores in predicted_data
                predicted_data[i]["highest_voted_valid"] = {
                    "predicted_sql": predicted_sql,
                    "predicted_execution": predicted_execution,
                }
                for metric, score in zip(metrics, metric_scores):
                    predicted_data[i]["highest_voted_valid"][f"{metric}_score"] = score
            else:
                raise Exception(f"Method {method} is not supported")

            # Store scores for each metric
            for metric, score in zip(metrics, metric_scores):
                all_methods[method][f"{metric}_scores"].append(score)
            all_methods[method]["valids"].append(has_valid)

    all_methods_meaned = {}
    for key, values in all_methods.items():
        if not key in all_methods_meaned:
            all_methods_meaned[key] = {}
        for eval_key, eval_values in values.items():
            all_methods_meaned[key][eval_key] = np.mean(eval_values) * 100

    return all_methods_meaned
