from txt2sql.metrics import soft_f1

from .match_utils import get_match_for_valid_exec


def df_soft_f1_score(df, results, config, out_col):
    execution_col = config.ex_label_col
    pred_execution_col = config.pred_ex_label_col
    df[out_col] = df.apply(
        lambda row: int(
            get_match_for_valid_exec(
                "soft_f1_score",
                soft_f1,
                row[pred_execution_col],
                row[execution_col],
                row["Fail"],
            ),
        ),
        axis=1,
    )
    true_count = df[out_col].sum().item()
    accuracy = (true_count / len(df)) * 100
    results["soft_f1_score"] = {"accuracy": accuracy}
    return df
