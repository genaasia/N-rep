from txt2sql.metrics import intent_match

from .match_utils import get_match_for_valid_exec


def df_intent_match(df, results, config, out_col):
    execution_col = config.ex_label_col
    pred_execution_col = config.pred_ex_label_col
    df[out_col] = df.apply(
        lambda row: get_match_for_valid_exec(
            "intent_match", intent_match, row[pred_execution_col], row[execution_col], row["Fail"]
        ),
        axis=1,
    )
    true_count = df[out_col].sum().item()
    false_count = len(df) - true_count
    accuracy = (true_count / (true_count + false_count)) * 100
    results["intent_match"] = {
        "matched": true_count,
        "different": false_count,
        "accuracy": accuracy,
    }
    return df
