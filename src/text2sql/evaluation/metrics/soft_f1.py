from .match_utils import get_match_for_valid_exec


def calculate_row_match(predicted_row, ground_truth_row):
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    for pred_val in predicted_row:
        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1
    for truth_val in ground_truth_row:
        if truth_val not in predicted_row:
            element_in_truth_only += 1
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    return match_percentage, pred_only_percentage, truth_only_percentage


def get_soft_f1_score(predicted, ground_truth):
    # if both predicted and ground_truth are empty, return 1.0 for f1_score
    if not predicted and not ground_truth:
        return 1.0

    # def remove_duplicates(lst):
    #     seen = set()
    #     return [d for d in lst if frozenset(d.items()) not in seen and not seen.add(frozenset(d.items()))]
    def remove_duplicates(list_of_dicts: list[dict]) -> list[dict]:
        return [d for i, d in enumerate(list_of_dicts) if str(d) not in str([list_of_dicts[:i]])]

    # Drop duplicates
    predicted_set = remove_duplicates(predicted)
    ground_truth_set = remove_duplicates(ground_truth)

    # convert back to list
    predicted = list(predicted_set)
    ground_truth = list(ground_truth_set)

    # Calculate matching scores for each possible pair
    match_scores = []
    pred_only_scores = []
    truth_only_scores = []

    for i, gt_row in enumerate(ground_truth):
        # rows only in the ground truth results
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        pred_row = predicted[i]
        match_score, pred_only_score, truth_only_score = calculate_row_match(pred_row, gt_row)
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)

    # rows only in the predicted results
    for i in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1_score


def df_soft_f1_score(df, results, config, out_col):
    execution_col = config.ex_label_col
    pred_execution_col = config.pred_ex_label_col
    df[out_col] = df.apply(
        lambda row: int(
            get_match_for_valid_exec(
                "soft_f1_score",
                get_soft_f1_score,
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
