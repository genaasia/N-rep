# example inference code snippets

complete and encapsulated inference and sqlite query code

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

## execute

see `example.ipynb` example for complete code.
