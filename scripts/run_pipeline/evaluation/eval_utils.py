from sql_metadata import Parser
from txt2sql.metrics import execution_match, intent_match, soft_f1, sql_match


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


def upper_bound_eval(predictions, target_sql, target_execution, score_cache, metrics=None):
    if metrics is None:
        metrics = ["execution_match"]
        
    has_valid = False
    predicted_valids = [item for item in predictions if item["valid"]]
    score_metrics = {metric: [] for metric in metrics}
    
    # Mapping of score types to their evaluation functions
    score_functions = {
        "sql_match": {
            "eval_func": sql_match,
            "stop_condition": any,
        },
        "execution_match": {
            "eval_func": execution_match,
            "stop_condition": any,
        },
        "intent": {
            "eval_func": intent_match,
            "stop_condition": any,
        },
        "soft_f1": {
            "eval_func": soft_f1,
            "stop_condition": lambda score_list: 1.0 in score_list,
        },
    }
    
    for values in predicted_valids:
        predicted_sql = values["sql"]
        if not values["valid"]:
            continue
        has_valid = True
        predicted_execution = values["results"]
        for score_name in metrics:
            if score_name not in score_functions:
                continue
            eval_func, stop_condition = score_functions[score_name]["eval_func"], score_functions[score_name]["stop_condition"]
            if not stop_condition(score_metrics[score_name]):
                score_metrics[score_name].append(
                    run_eval_with_cache(
                        predicted_sql,
                        target_sql,
                        predicted_execution if score_name != "sql_match" else predicted_sql,
                        target_execution if score_name != "sql_match" else target_sql,
                        eval_func,
                        score_cache,
                    )
                )
    
    return tuple([any(score_metrics[metric]) for metric in metrics] + [has_valid])


def lower_bound_eval(predictions, target_sql, target_execution, score_cache, metrics=None):
    if metrics is None:
        metrics = ["execution_match"]
        
    if not all([item["valid"] for item in predictions]):
        return tuple([False] * len(metrics) + [False])
        
    score_metrics = {metric: [] for metric in metrics}
    
    # Mapping of score types to their evaluation functions
    score_functions = {
        "sql_match": {
            "eval_func": sql_match,
            "calculate_condition": all,
        },
        "execution_match": {
            "eval_func": execution_match,
            "calculate_condition": all,
        },
        "intent": {
            "eval_func": intent_match,
            "calculate_condition": all,
        },
        "soft_f1": {
            "eval_func": soft_f1,
            "calculate_condition": lambda score_list: not 0.0 in score_list,
        },
    }
    
    for values in predictions:
        predicted_execution = values["results"]
        predicted_sql = values["sql"]
        for score_name in metrics:
            if score_name not in score_functions:
                continue
            eval_func, calculate_condition = score_functions[score_name]["eval_func"], score_functions[score_name]["calculate_condition"]
            if calculate_condition(score_metrics[score_name]):
                score_metrics[score_name].append(
                    run_eval_with_cache(
                        predicted_sql,
                        target_sql,
                        predicted_execution if score_name != "sql_match" else predicted_sql,
                        target_execution if score_name != "sql_match" else target_sql,
                        eval_func,
                        score_cache,
                    )
                )
    
    return tuple([all(score_metrics[metric]) for metric in metrics] + [True])


def highest_voted_eval(predictions, target_sql, target_execution, score_cache, metrics=None):
    if metrics is None:
        metrics = ["execution_match"]
        
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
        return tuple([False] * len(metrics) + [False])

    predicted_execution = values["results"]

    score_functions = {
        "sql_match": sql_match,
        "execution_match": execution_match,
        "intent": intent_match,
        "soft_f1": soft_f1,
    }
    
    scores = []
    for metric in metrics:
        if metric not in score_functions:
            continue
        eval_func = score_functions[metric]
        scores.append(
            run_eval_with_cache(
                predicted_sql,
                target_sql,
                predicted_execution if metric != "sql_match" else predicted_sql,
                target_execution if metric != "sql_match" else target_sql,
                eval_func,
                score_cache,
            )
        )
    return tuple(scores + [True])


def highest_voted_valid_eval(predictions, target_sql, target_execution, score_cache, metrics=None):
    if metrics is None:
        metrics = ["execution_match"]
        
    prediced_valids = [item for item in predictions if item["valid"]]
    if not prediced_valids:
        return tuple([False] * len(metrics) + [False, None, None])

    prediction_votes = {}
    for values in prediced_valids:
        prediction = values["sql"]
        found_match = False
        for compared_prediction, compared_values in prediction_votes.items():
            if execution_match(compared_values["results"], values["results"]):
                found_match = True
                prediction_votes[compared_prediction]["vote_count"] += 1
                break
        if not found_match:
            prediction_votes[prediction] = {"vote_count": 0, **values}

    predicted_sql, values = max(
        prediction_votes.items(), key=lambda x: x[1]["vote_count"]
    )
    predicted_execution = values["results"]

    score_functions = {
        "sql_match": sql_match,
        "execution_match": execution_match,
        "intent": intent_match,
        "soft_f1": soft_f1,
    }
    
    scores = []
    for metric in metrics:
        if metric not in score_functions:
            continue
        eval_func = score_functions[metric]
        scores.append(
            run_eval_with_cache(
                predicted_sql,
                target_sql,
                predicted_execution if metric != "sql_match" else predicted_sql,
                target_execution if metric != "sql_match" else target_sql,
                eval_func,
                score_cache,
            )
        )
    return tuple(scores + [True, predicted_sql, predicted_execution])
