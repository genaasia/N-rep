# example inference code snippets

complete and encapsulated inference and evaluation code.

## setup

### download and extract the BIRD dataset

in my setup, BIRD dev file is extracted here: `/data/sql_datasets/bird/dev_20240627`

and the dev databases are extracted inside, to: `/data/sql_datasets/bird/dev_20240627/dev_databases`

so it looks like

```
(base) derek@jennifer:/data/sql_datasets/bird/dev_20240627/dev_databases$ ls
california_schools       european_football_2  superhero
card_games               financial            thrombosis_prediction
codebase_community       formula_1            toxicology
debit_card_specializing  student_club

```

### create ENV file

create a new file called `.env` and add the following content (replacing with actual values from azure):

```
AZURE_OPENAI_API_KEY="abcd1234wxyz7890hello"
AZURE_OPENAI_ENDPOINT="https://my-workspace.openai.azure.com/"
```

NOTE: gpt-4o is in "gena-gpt-2" and o3 is in "gena-gpt-fine-tuning"

### install requirements

run `pip install requirements.txt`

## run inference

run inference like below.

- `--database-directory` should point to the BIRD dev_databases location
- `--model-name` should be "gena-4o-2024-08-06" (recommended 4o), "gena-4o" or "gena-o3-mini". set the env file url accordingly!
- `--threads` controls parallel requests. maybe set it lower (1~2) for o3 due to usage limits

it will output two files in the same path as the input file:

- (input_file)_messages.json -> the messages input to the LLM, for debugging
- (input_file)_predictions.json -> the input file, with new "predicted_SQL" key

```
python run_inference.py \
  --input-file ./data/bird_dev_augmented_sample.json \
  --database-directory  /data/sql_datasets/bird/dev_20240627/dev_databases \
  --model-name gena-4o \
  --temperature 0.0 \
  --threads 4
```

## chunked inference

if you run into timeout errors, you can use the `split_files.py` script to split the test data into n-row chunks.

then, you can use the `run_inference_batched.sh` script to process each file individually, with sleep between chunks.

(you should edit the sh file as you must specify the target directory and your conda environment inside!)

## run evaluation

now you can run evaluation on the predictions.

pass in the predictions from the previous step, and it will output a CSV with all information.

```
python run_evaluation.py \
  --prediction-file ./data/bird_dev_augmented_sample_gena-4o_predictions.json \
  --database-directory  /data/sql_datasets/bird/dev_20240627/dev_databases \
  --threads 4
```

when the script finishes, you will get a summary printed to the terminal:

```
Evaluation summary:
Soft F1        : 0.357
Intent Match   : 0.537
Execution Match: 0.500
```

but check the saved CSV file for row-by-row information