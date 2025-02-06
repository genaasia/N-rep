# text2sql_testbed

module-based code for running text2sql tests

## development progress

- [x] data access:
    - [x] sqlite data loader
    - [x] mysql data connector
    - [x] mysql data connector
- [ ] preprocessors:
    - [ ] pinterest style query summarizer
    - [ ] pinterest style table summarizer
- [x] embeddings:
    - [x] azure embedder
    - [x] bedrock cohere embedder
    - [x] bedrock titan v2 embedder
    - [x] `sentence-transformers` embedder
    - [ ] (*later*) cohere embedder
    - [ ] (*later*) azure async embedder
- [x] retrieval:
    - [x] local retriever
    - [x] weaviate retiever
- [x] schema formatter:
    - [x] sqlite formatter
    - [x] mysql formatter
    - [x] postgres formatter
- [ ] prompt formatter:
    - [x] basic few-shot conversation formatter
    - [x] sql markdown code block formatter
    - [ ] alpaca in-context learning logic (like DAIL-SQL)
    - [x] few-shot message logic (like Dubo-SQL)
- [ ] generation:
    - [x] azure generator
    - [x] bedrock generator
    - [ ] (*later*) sagemaker (fine-tuned) generator
    - [ ] (*later*) pandas style inference mode (*Before Generation, Align it!*)
- [ ] postprocessing:
    - [x] validation
    - [ ] repair prompt
- [ ] inference
    - [x] experiment config parser and runner
    - [ ] pre-SQL-generator (*Before Generation, Align it!*)
- [ ] evaluation
    - [x] port evaluation code from gena-ai
    - [ ] `Evaluator` class

## development questions

- should we use `pydantic` schema for outputs? e.g. retrieval outputs?
- how to version, format prompts?
- want to support completion-like as well as conversation LLM APIs?

## development notes

use `git flow` process *and* tool for preparing releases (merging dev to main)!

1. create feature branches from `dev` with name `feature/<description>`
2. merge branches back into `dev` via Github Pull Request (use notion ticket tag in title! "[GHT-XXX] my pr title")
3. for preparing numbered releases, use `git-flow` and merge from `dev` to `main` via `release/x.x.x` branch
4. before `finish` ing the release, bump the `version.py` number!
5. use the `git push origin --tags` to update the version info on GitHub!
6. also need to `push` local changes on *both* `main` and `dev`
7. go to github repo page -> releases -> releases -> draft new release -> select tag -> autogenerate docs

see the `git-flow` [cheatsheet](http://danielkummer.github.io/git-flow-cheatsheet/) for help.

## environment

### start dependencies

for weaviate `Retriever`, you need a running instance of weaviate.

(*todo*) see docker compose file for details.

### setup from files

you can setup your environment with either `conda` or `pip` (*conda is recommended*).

(derek) i am using `conda` so the `pip` requirements file may not be perfect, see comments from references.

**conda**

you can just run the `conda env update` line to update an existing environment

```
conda create --name text2sql python=3.12 pip
conda activate text2sql
conda env update --file requirements.yaml --prune
``` 

note: you can use `--name <env>` flag to update a conda env without activating it.

**pip**

`pip install -r requirements.txt`

### update files

if you installed packages, you should update the requirements:

**script**

1. activate conda environment "text2sql"
2. run `./requirements_export.sh` to generate both files

**conda**

`conda env export > requirements.yaml`

**pip**

`pip list --format=freeze > requirements.txt`

### references

[From conda create requirements.txt for pip3](https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3)  
[How to update an existing Conda environment with a .yml file](https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file)

## how to use

final experiment script pending completion of all modules

to see how to use individual models, for now, check the notebooks.

you will need an `.env` file like below:

```
AZURE_OPENAI_API_KEY="<see azure openai dashboard for key>"
AZURE_OPENAI_API_ENDPOINT="https://gena-gpt-2.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-06-01"
AZURE_OPENAI_MODEL="gena-text-embedding-3-small"
AZURE_OPENAI_GEN_MODEL="gena-4o"

AWS_ACCESS_KEY_ID="<your aws access key>"
AWS_SECRET_ACCESS_KEY="<your aws secret key>"
```
