import json
import os
import argparse
from collections import defaultdict

# Path to ground truth file
gt_fn = "./schema_linking_gt.json"

def load_gt(gt_fn):
    with open(gt_fn, 'r') as f:
        gt_data = json.load(f)
    gt_map = {}
    for entry in gt_data:
        qid = entry['question_id']
        gt_map[qid] = entry['gold_table_map']
    return gt_map

def get_schema_from_pred(pred):
    # Try to get schema from 'table_description' or 'full_description' in prediction file
    schema = {}
    if 'table_description' in pred:
        try:
            desc = json.loads(pred['table_description'])
            for t, tdesc in desc['tables'].items():
                schema[t] = list(tdesc['columns'].keys())
        except Exception:
            pass
    if not schema and 'full_description' in pred:
        try:
            desc = json.loads(pred['full_description'])
            for t, tdesc in desc['tables'].items():
                schema[t] = list(tdesc['columns'].keys())
        except Exception:
            pass
    return schema

def expand_keep_all(pred_cols, schema):
    out = {}
    for t, cols in pred_cols.items():
        if cols == "keep_all":
            out[t] = set(schema.get(t, []))
        else:
            out[t] = set(cols)
    return out

def expand_keep_all_gt(gt_cols):
    # For gold, just use the columns listed in gold_table_map
    return {t: set(cols) for t, cols in gt_cols.items()}

def parse_variant_name(filename):
    # Extracts the part before _qid-xxxx.json
    base = os.path.basename(filename)
    if '_qid-' in base:
        return base.split('_qid-')[0]
    return base

def evaluate(pred_dir):
    gt_map = load_gt(gt_fn)
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json') and '_qid-' in f]
    # Group files by variant name
    variant_files = defaultdict(dict)  # variant_name -> {qid: filepath}
    for f in pred_files:
        try:
            qid = int(f.split('_qid-')[-1].split('.')[0])
            variant = parse_variant_name(f)
            variant_files[variant][qid] = os.path.join(pred_dir, f)
        except Exception:
            continue

    for variant, pred_map in variant_files.items():
        table_tp = 0
        table_fp = 0
        table_fn = 0
        col_tp = 0
        col_fp = 0
        col_fn = 0
        n_eval = 0

        for qid, gt_tables in gt_map.items():
            if qid not in pred_map:
                continue
            with open(pred_map[qid], 'r') as f:
                pred = json.load(f)
            schema = get_schema_from_pred(pred)
            pred_cols = pred.get('column_linking', {})
            # Handle null column_linking: treat as keep_all for all tables in schema
            if pred_cols is None:
                pred_cols = {t: 'keep_all' for t in schema.keys()}
            pred_cols = expand_keep_all(pred_cols, schema)
            gt_cols = expand_keep_all_gt(gt_tables)

            # Table-level
            pred_tables = set(pred_cols.keys())
            gold_tables = set(gt_cols.keys())
            tp_tables = pred_tables & gold_tables
            table_tp += len(tp_tables)
            table_fp += len(pred_tables - gold_tables)
            table_fn += len(gold_tables - pred_tables)

            # Column-level (micro)
            for t in gold_tables | pred_tables:
                pred_set = pred_cols.get(t, set())
                gold_set = gt_cols.get(t, set())
                tp = len(pred_set & gold_set)
                fp = len(pred_set - gold_set)
                fn = len(gold_set - pred_set)
                col_tp += tp
                col_fp += fp
                col_fn += fn
            n_eval += 1

        table_recall = table_tp / (table_tp + table_fn) if (table_tp + table_fn) > 0 else 0.0
        table_prec = table_tp / (table_tp + table_fp) if (table_tp + table_fp) > 0 else 0.0
        col_recall = col_tp / (col_tp + col_fn) if (col_tp + col_fn) > 0 else 0.0
        col_prec = col_tp / (col_tp + col_fp) if (col_tp + col_fp) > 0 else 0.0

        print(f"=== Variant: {variant} ===")
        print(f"Evaluated {n_eval} examples.")
        print(f"Table recall: {table_recall:.3f}, Table precision: {table_prec:.3f}")
        print(f"Column recall: {col_recall:.3f}, Column precision: {col_prec:.3f}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate schema linking predictions.")
    parser.add_argument("pred_dir", type=str, help="Directory with _qid-xxxx.json prediction files.")
    args = parser.parse_args()
    evaluate(args.pred_dir)