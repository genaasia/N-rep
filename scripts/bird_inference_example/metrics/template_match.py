from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from .match_utils import get_match_for_valid_exec


def get_template_match(pred_template, label_template):
    return label_template == pred_template


def df_template_match(df, results, config, out_col):
    template_id_col = config.template_id_col
    pred_template_id_col = config.pred_template_id_col
    df[out_col] = df.apply(
        lambda row: get_match_for_valid_exec(
            "template_match",
            get_template_match,
            row[pred_template_id_col],
            row[template_id_col],
            row["Fail"],
        ),
        axis=1,
    )
    true_count = df[out_col].sum().item()
    false_count = len(df) - true_count

    templates = df[template_id_col].to_list()
    pred_templates = df[pred_template_id_col].to_list()
    precision = precision_score(templates, pred_templates, average="macro") * 100
    recall = recall_score(templates, pred_templates, average="macro") * 100
    f1 = f1_score(templates, pred_templates, average="macro") * 100
    acc = accuracy_score(templates, pred_templates) * 100

    results["template_match"] = {
        "matched": true_count,
        "different": false_count,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return df
