import argparse
import json
import os


# parse arguments
parser = argparse.ArgumentParser("convert gena model output json to tab-separated eval format")
parser.add_argument('--original-file', type=str, default=None, help='path to source file (e.g. dev.json)')
parser.add_argument('--source-file', type=str, required=True, help='path to gena prediction json file')
parser.add_argument('--target-file', type=str, required=True, help='path to new bird eval compatible sql file (e.g. predicted_dev.json)')
parser.add_argument('--question_id_key', type=str, default='question_id', help='question_id key name, default question_id')
parser.add_argument('--db_id_key', type=str, default='db_id', help='db_id key name, default db_id')
parser.add_argument('--prediction_key', type=str, default='predictions', help='predictions key name, default predictions')
parser.add_argument('--sql_key', type=str, default='sql', help='predictions sql key name, default sql')
args = parser.parse_args()


# check input is json file
assert args.original_file.endswith('.json'), "original file must be a json file"
assert args.source_file.endswith('.json'), "source file must be a json file"
assert args.target_file.endswith('.json'), "target file must be a json file"
out = os.path.basename(args.target_file).replace('.json', '')
assert out.startswith("predict_"), "target file must start with 'predict_'"


# create output directory, if not exists
output_dir = os.path.dirname(args.target_file)
if not os.path.exists(output_dir):
    print(f"creating output directory '{output_dir}'")
    os.makedirs(output_dir)


# read in original data from json
print(f"loading original data from '{args.original_file}'")
with open(args.source_file, "r") as f:
    predicted_data = json.load(f)
# sort by question_id for later, serves as basic check as well
predicted_data = sorted(predicted_data, key=lambda x: x[args.question_id_key])
# validate that sql and db_id are present
for i, row in enumerate(predicted_data):
    assert args.prediction_key in row, f"missing sql key ` in row {i}"
    if len(row[args.prediction_key]) > 0:
        assert args.sql_key in row[args.prediction_key][0], f"missing sql in row {i}"
    assert args.db_id_key in row, f"missing db_id in row {i}"
print(f"loaded {len(predicted_data)} predictions from '{os.path.basename(args.source_file)}'")


# load original data and check, if supplied
if args.original_file:
    print(f"loading original data from '{args.original_file}'")
    with open(args.original_file, 'r') as f:
        original_data = json.load(f)
    if len(predicted_data) != len(original_data):
        print(f"Warning: original ({len(original_data)}) and predicted data ({len(predicted_data)}) are different lengths! Will align with question_id and sub missing with empty strings.")
    if set([x[args.question_id_key] for x in predicted_data]) != set([x[args.question_id_key] for x in original_data]):
        print(f"Warning: original and predicted data have different question_ids! Will align with original question_id and sub missing with empty strings.")
    target_question_ids = [x[args.question_id_key] for x in original_data]
    print(f"confirmed predicted data length and question_ids against original file {os.path.basename(args.original_file)}")
else:
    print("warning, no original file supplied, will only convert prediction data as-is!")
    target_question_ids = [x[args.question_id_key] for x in predicted_data]


# convert prediction file to target format
print(f"converting prediction data to target format...")
output_dict = {}
# actually, convert to dict for faster retrieval
predicted_data_dict = {x[args.question_id_key]: x for x in predicted_data}
for qid in target_question_ids:
    # get the predicted SQL from the prediction data, else empty string " "
    sample = predicted_data_dict.get(qid, dict())
    db_id = sample[args.db_id_key]
    predictions = sample.get(args.prediction_key, list())
    if len(predictions) < 1:
        output_dict[str(qid)] = f" \t----- bird -----\t{db_id}"
        continue
    prediction = predictions[0]
    predicted_sql = prediction.get(args.sql_key, " ").replace("\n", " ")
    # insert the formatted string
    output_dict[str(qid)] = f"{predicted_sql}\t----- bird -----\t{db_id}"
print(f"converted {len(output_dict)} predictions to target format")


with open(args.target_file, 'w') as f:
    json.dump(output_dict, f, indent=2)
print(f"saved converted predictions to '{args.target_file}'")
print("all done!")