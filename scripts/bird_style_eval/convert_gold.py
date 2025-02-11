import argparse
import json
import os


# parse arguments
parser = argparse.ArgumentParser("convert BIRD json to tab-separated eval format")
parser.add_argument('--original-file', type=str, default=None, help='path to source file (e.g. dev.json)')
parser.add_argument('--target-file', type=str, required=True, help='path to new bird eval compatible json file')
parser.add_argument('--question_id_key', type=str, default='question_id', help='question_id key name, default question_id')
parser.add_argument('--db_id_key', type=str, default='db_id', help='db_id key name, default db_id')
parser.add_argument('--sql_key', type=str, default='SQL', help='sql key name, default SQL')
args = parser.parse_args()


# check input is json file
assert args.original_file.endswith('.json'), "original file must be a json file"
# check output file basename == inputfile.json -> inputfile_gold.sql
assert args.target_file.endswith('.sql'), "target file must be a sql file"
inp = os.path.basename(args.original_file).replace('.json', '')
out = os.path.basename(args.target_file).replace('.sql', '')
assert inp + "_gold" == out, f"target file {out}.sql must be named {inp}_gold.sql"

# load dataset
print(f"loading original data from '{args.original_file}'")
with open(args.original_file, 'r') as f:
    original_data = json.load(f)
    original_data = sorted(original_data, key=lambda x: x[args.question_id_key])
print(f"loaded {len(original_data)} original data from '{os.path.basename(args.original_file)}'")


# write to sql-dbname tab-split format
print(f"writing gold fmt to '{args.target_file}'")
with open(args.target_file, 'w') as f:
    for idx, row in enumerate(original_data):
        sql = row[args.sql_key]
        db_id = row[args.db_id_key]
        if idx != 0:
            f.write("\n")
        f.write(f"{sql}\t{db_id}")
print(f"done writing gold fmt to '{args.target_file}'")
print("done!")