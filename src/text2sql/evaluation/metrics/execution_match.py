import math
from dataclasses import dataclass

from .match_utils import get_match_for_valid_exec


@dataclass(frozen=True)
class ComparableTuple:
    data: tuple

    def __eq__(self, other):
        if len(self.data) != len(other.data):
            return False

        for v1, v2 in zip(self.data, other.data):
            if isinstance(v1, float) and isinstance(v2, float):
                # Whole point of this clase is so that we can do set comparison
                # while handling floating-point representation errors
                if not math.isclose(v1, v2, rel_tol=1e-9):
                    return False
            elif v1 != v2:
                return False
        return True

    def __hash__(self):
        # Convert floats to strings with limited precision for hashing
        processed = tuple(f"{x:.10f}" if isinstance(x, float) else x for x in self.data)
        return hash(processed)


def get_execution_match(pred_ex, label_ex):
    def transform_results(ex_result):
        new_ex_result = []
        for row in ex_result:
            new_row = ComparableTuple(tuple(row.values()))
            new_ex_result.append(new_row)
        return set(new_ex_result)

    return transform_results(label_ex) == transform_results(pred_ex)


def df_execution_match(df, results, config, out_col):
    execution_col = config.ex_label_col
    pred_execution_col = config.pred_ex_label_col
    df[out_col] = df.apply(
        lambda row: get_match_for_valid_exec(
            "execution_match",
            get_execution_match,
            row[pred_execution_col],
            row[execution_col],
            row["Fail"],
        ),
        axis=1,
    )
    true_count = df[out_col].sum().item()
    false_count = len(df) - true_count
    accuracy = (true_count / (true_count + false_count)) * 100
    results["execution_match"] = {
        "matched": true_count,
        "different": false_count,
        "accuracy": accuracy,
    }
    return df
