from sql_metadata import Parser
from text2sql.evaluation.metrics import (
    get_execution_match,
    get_intent_match,
    get_soft_f1_score,
    get_sql_match,
)


def get_top_voted_queries(queries_list, n):
    sorted_queries = sorted(
        queries_list.items(), key=lambda x: x[1]["vote_count"], reverse=True
    )
    return sorted_queries[:n]


def run_eval_with_cache(
    predicted_sql, target_sql, prediction, target, eval_func, cache
):
    cache_key = (predicted_sql, target_sql)
    eval_name = eval_func.__name__
    if cache_key not in cache:
        cache[cache_key] = {}
    if eval_name not in cache[cache_key]:
        cache[cache_key][eval_name] = eval_func(prediction, target)
    return cache[cache_key][eval_name]


def prune_schema(schema_description, candidates):
    c_tables = []
    for candidate in candidates:
        c_tables = c_tables + Parser(candidate).tables

    tables = schema_description.strip().split("\n")
    # Dictionary to store the results
    tables_dict = {}
    for table in tables:
        # Skip empty lines
        if not table.strip():
            continue
        # Find text between 'table '
        table_name = table.split("'")[1]
        tables_dict[table_name.lower()] = table

    new_schema = {}
    for table_name in c_tables:
        if table_name.lower() in tables_dict:
            new_schema[table_name.lower()] = tables_dict[table_name.lower()]
    new_schema = "\n".join(new_schema.values())
    return new_schema


def upper_bound_eval(predictions, target_sql, target_execution, score_cache):
    has_valid = False
    predicted_valids = [item for item in predictions if item["valid"]]
    score_metrics = {
        "sql_match": [],
        "execution_match": [],
        "intent": [],
        "soft_f1": [],
    }
    # Mapping of score types to their evaluation functions
    score_functions = {
        "sql_match": {
            "eval_func": get_sql_match,
            "stop_condition": any,
        },
        "execution_match": {
            "eval_func": get_execution_match,
            "stop_condition": any,
        },
        "intent": {
            "eval_func": get_intent_match,
            "stop_condition": any,
        },
        "soft_f1": {
            "eval_func": get_soft_f1_score,
            "stop_condition": lambda score_list: 1.0 in score_list,
        },
    }
    for values in predicted_valids:
        predicted_sql = values["sql"]
        if not values["valid"]:
            continue
        has_valid = True
        predicted_execution = values["results"]
        for score_name, vals in score_functions.items():
            eval_func, stop_condition = vals["eval_func"], vals["stop_condition"]
            if not stop_condition(score_metrics[score_name]):
                score_metrics[score_name].append(
                    run_eval_with_cache(
                        predicted_sql,
                        target_sql,
                        (
                            predicted_execution
                            if score_name != "sql_match"
                            else predicted_sql
                        ),
                        target_execution if score_name != "sql_match" else target_sql,
                        eval_func,
                        score_cache,
                    )
                )
    return (
        any(score_metrics["sql_match"]),
        any(score_metrics["execution_match"]),
        any(score_metrics["intent"]),
        max(score_metrics["soft_f1"]) if score_metrics["soft_f1"] else 0.0,
        has_valid,
    )


def lower_bound_eval(predictions, target_sql, target_execution, score_cache):
    if not all([item["valid"] for item in predictions]):
        return False, False, False, 0.0, False
    score_metrics = {
        "sql_match": [],
        "execution_match": [],
        "intent": [],
        "soft_f1": [],
    }
    # Mapping of score types to their evaluation functions
    score_functions = {
        "sql_match": {
            "eval_func": get_sql_match,
            "calculate_condition": all,
        },
        "execution_match": {
            "eval_func": get_execution_match,
            "calculate_condition": all,
        },
        "intent": {
            "eval_func": get_intent_match,
            "calculate_condition": all,
        },
        "soft_f1": {
            "eval_func": get_soft_f1_score,
            "calculate_condition": lambda score_list: not 0.0 in score_list,
        },
    }
    for values in predictions:
        predicted_execution = values["results"]
        predicted_sql = values["sql"]
        for score_name, vals in score_functions.items():
            eval_func, calculate_condition = (
                vals["eval_func"],
                vals["calculate_condition"],
            )
            if calculate_condition(score_metrics[score_name]):
                score_metrics[score_name].append(
                    run_eval_with_cache(
                        predicted_sql,
                        target_sql,
                        (
                            predicted_execution
                            if score_name != "sql_match"
                            else predicted_sql
                        ),
                        target_execution if score_name != "sql_match" else target_sql,
                        eval_func,
                        score_cache,
                    )
                )
    return (
        all(score_metrics["sql_match"]),
        all(score_metrics["execution_match"]),
        all(score_metrics["intent"]),
        min(score_metrics["soft_f1"]) if score_metrics["soft_f1"] else 0.0,
        True,
    )


def highest_voted_eval(predictions, target_sql, target_execution, score_cache):
    prediction_votes = {}
    for values in predictions:
        prediction = values["sql"]
        if prediction not in prediction_votes:
            prediction_votes[prediction] = {"vote_count": 0, **values}
        prediction_votes[prediction]["vote_count"] += 1

    predicted_sql, values = max(
        prediction_votes.items(), key=lambda x: x[1]["vote_count"]
    )
    if not values["valid"]:
        return False, False, False, 0.0, False

    predicted_execution = values["results"]

    eval_functions = [
        get_sql_match,
        get_execution_match,
        get_intent_match,
        get_soft_f1_score,
    ]
    scores = []
    for eval_func in eval_functions:
        scores.append(
            run_eval_with_cache(
                predicted_sql,
                target_sql,
                predicted_execution if eval_func != get_sql_match else predicted_sql,
                target_execution if eval_func != get_sql_match else target_sql,
                eval_func,
                score_cache,
            )
        )
    return *scores, True


def highest_voted_valid_eval(predictions, target_sql, target_execution, score_cache):
    prediced_valids = [item for item in predictions if item["valid"]]
    if not prediced_valids:
        return False, False, False, 0.0, False, None, None

    prediction_votes = {}
    for values in prediced_valids:
        prediction = values["sql"]
        if prediction not in prediction_votes:
            prediction_votes[prediction] = {"vote_count": 0, **values}
        prediction_votes[prediction]["vote_count"] += 1

    predicted_sql, values = max(
        prediction_votes.items(), key=lambda x: x[1]["vote_count"]
    )
    predicted_execution = values["results"]

    get_intent_normalized = lambda x, y: get_intent_match(x, y, normalize_dates=True)
    eval_functions = [
        get_sql_match,
        get_execution_match,
        get_intent_normalized,
        get_soft_f1_score,
    ]
    scores = []
    for eval_func in eval_functions:
        scores.append(
            run_eval_with_cache(
                predicted_sql,
                target_sql,
                predicted_execution if eval_func != get_sql_match else predicted_sql,
                target_execution if eval_func != get_sql_match else target_sql,
                eval_func,
                score_cache,
            )
        )
    return *scores, True, predicted_sql, predicted_execution
