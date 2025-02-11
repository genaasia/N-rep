# BIRD style evaluation

## license

`evaluation.py` comes from https://github.com/AlibabaResearch/DAMO-ConvAI.

it is used here under MIT license and original license file is included.

## instructions

requirements: see `requirements.txt`

### convert source file

convert BIRD json file to BIRD-eval-compatible ground truth file.

for train, there is "train.json" and "train_gold.sql". for dev, there is only "dev.json".

```
python convert_gold.py \
  --original-file /path/to/bird-dev/dev.json \
  --target-file /path/to/bird-dev/dev_gold.sql
```

example:

```
python convert_gold.py \
  --original-file /data/sql_datasets/bird/dev_20240627/dev.json \
  --target-file /data/sql_datasets/bird/dev_20240627/dev_gold.sql
```

### convert gena output json

convert gena verbose output JSON file to BIRD-eval-compatible predictions file.

the file must be called "predict_xxx.json", where xxx -- e.g. "train" or "dev", so separate with subdirectories.

```
python convert.py \
  --original-file /path/to/bird-dev/dev.json \
  --source-file /path/to/gena/output/file.json \
  --target-file /path/to/bird-eval/compatible.json
```

example:

```
python convert.py \
  --original-file /data/sql_datasets/bird/dev_20240627/dev.json \
  --source-file /home/derek/PythonProjects/gena/_experiment_/text2sql_testbed/scripts/run_pipeline/inference/bird_dev/original/pipe_v1_basic_types_relations_2025-02-06--22-34-13.json \
  --target-file ./inputs/original/predict_dev.json
```


### run bird-eval

modified: add tqdm progress bar and save results to file using `--out_fpath`

suggest to set out_fpath directory to as same input predicted sql path.

it will save the output file, and also the exec_result.json dump.

note: use the final backslashes on paths or the filepaths will not be correct!!!

```
python evaluation.py \
  --predicted_sql_path <same as base bath of --target-file above, w/o filename> \
  --ground_truth_path <path to bird dev_20240627> \
  --data_mode dev \
  --db_root_path <path to bird dev_20240627/dev_databases> \
  --num_cpus 2 \
  --mode_gt gt \
  --mode_predict gpt \
  --diff_json_path <path to bird dev_20240627/dev.json> \
  --meta_time_out 30.0 \
  --out_fpath <new output file name>
```

example:

```
python evaluation.py \
  --predicted_sql_path ./inputs/gold/ \
  --ground_truth_path /data/sql_datasets/bird/dev_20240627/ \
  --data_mode dev \
  --db_root_path /data/sql_datasets/bird/dev_20240627/dev_databases/ \
  --num_cpus 2 \
  --mode_gt gt \
  --mode_predict gpt \
  --diff_json_path /data/sql_datasets/bird/dev_20240627/dev.json \
  --meta_time_out 30.0 \
  --out_fpath ./inputs/gold/results.txt
```

### merge results to dataframe

combines the gena json data and the execution accuracy results.

also, includes rapidfuzz string matching `token set ratio` for source and synthesized question, as well as between target and output SQL, as further heuristics for examination.

```
python export.py \
  --source-file <path to gena output json> \
  --target-path <--out_fpath value base directory>
```

example:

```
python export.py \
  --source-file /home/derek/PythonProjects/gena/_experiment_/text2sql_testbed/scripts/run_pipeline/inference/bird_dev/original/pipe_v1_basic_types_relations_2025-02-06--22-34-13.json \
  --target-path ./inputs/original
```