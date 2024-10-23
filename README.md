# text2sql_testbed

module-based code for running text2sql tests

## development progress

- [ ] data access:
    - [ ] sqlite data loader
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
- [ ] schema formatter:
    - [ ] sqlite formatter
    - [ ] mysql formatter
    - [ ] postgres formatter
- [ ] generation:
    - [ ] alpaca in-context learning logic (like DAIL-SQL)
    - [ ] few-shot message logic (like Dubo-SQL)
    - [ ] bedrock generator
    - [ ] sagemaker (fine-tuned) generator
    - [ ] azure generator
- [ ] postprocessing:
    - [ ] validation
    - [ ] repair prompt

## development questions

- should we use `pydantic` schema for outputs? e.g. retrieval outputs

## environment

### start dependencies

for weaviate retrieval, you need a running instance of weaviate.

see the 

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