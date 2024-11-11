import decimal

from .match_utils import get_match_for_valid_exec


def get_intent_match(actual, expected):
    # Function to remove leading and trailing quotes
    def remove_quotes(string):
        if isinstance(string, str):
            return string.strip('"')
        return string

    # Convert all keys in expected and actual to lowercase for case-insensitive comparison
    expected = [{(k.lower() if k is not None else None): v for k, v in row.items()} for row in expected]
    actual = [{(k.lower() if k is not None else None): v for k, v in row.items()} for row in actual]

    # Remove quotes in string values in expected and actual
    expected = [{k: remove_quotes(v) for k, v in row.items()} for row in expected]
    actual = [{k: remove_quotes(v) for k, v in row.items()} for row in actual]

    if len(expected) != len(actual):
        # print(f"Number of rows are different")
        return False

    def is_match(expected_val, actual_val):
        """
        This function compares expected_val and actual_val with relevant datatype conversion
        """
        if (expected_val is None or expected_val == 0) and (actual_val is None or actual_val == 0):
            return True
        elif isinstance(expected_val, (int, float, decimal.Decimal)) and isinstance(
            actual_val, (int, float, decimal.Decimal)
        ):
            return actual_val is not None and (str(round(expected_val, 1)) == str(round(actual_val, 1)))
        elif isinstance(expected_val, str) and isinstance(actual_val, str):
            return str(actual_val) == expected_val
        elif isinstance(expected_val, str) and isinstance(actual_val, (int, float)):
            return expected_val == str(actual_val)
        elif isinstance(expected_val, (int, float)) and isinstance(actual_val, str):
            return str(expected_val) == actual_val
        elif isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
            return (actual_val is not None) and (round(expected_val, 0) == round(actual_val, 0))
        else:
            return str(expected_val) == str(actual_val)

    def match_rule(expected_row, actual_row):
        # columns_to_compare = expected_row.keys()
        for column in expected_row.keys():
            if column in actual_row:
                if not is_match(expected_row[column], actual_row[column]):
                    return False
            else:
                column_matched = any(is_match(expected_row[column], actual_val) for actual_val in actual_row.values())
                if not column_matched:
                    return False
        return True

    for expected_row in expected:
        row_matched = False
        for actual_row in actual:
            actual_row_copy = actual_row.copy()
            if match_rule(expected_row, actual_row_copy):
                row_matched = True
                break
            # if row_matched:
            #     matched = True
            #     actual.remove(actual_row)
            #     break
        if not row_matched:
            # print(f"Comparison failed for row: exp={expected_row}, act={actual_row}")
            return False
    return True


def df_intent_match(df, results, config, out_col):
    execution_col = config.ex_label_col
    pred_execution_col = config.pred_ex_label_col
    df[out_col] = df.apply(
        lambda row: get_match_for_valid_exec(
            "intent_match", get_intent_match, row[pred_execution_col], row[execution_col], row["Fail"]
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
