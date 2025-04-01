# insert data

## env file

```
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
WEAVIATE_API_KEY=""
```

## embed

example

```
python embed.py \
  --input-file /data/gena_data/embeddings/menswear/menswear_train_nl2sql_20241227_synthesized_translated_addbusinessquestions.csv \
  --output-path /data/gena_data/embeddings/menswear \
  --nl-question english_query
```

## upload

example

```
python populate.py \
  --input-file /data/gena_data/embeddings/menswear/menswear_train_nl2sql_20241227_synthesized_translated_addbusinessquestions.csv \
  --embedding-file /data/gena_data/embeddings/menswear/menswear_train_nl2sql_20241227_synthesized_translated_addbusinessquestions/english_query_bedrock-cohere_cohere.embed-multilingual-v3.npy \
  --use-weaviate-cloud \
  --weaviate-collection-name Test_collection \
  --weaviate-cluster-url=https://aoylzn0zsqgedk2i48djua.c0.asia-southeast1.gcp.weaviate.cloud \
  --nl-question english_query \
  --sql-template sql_template \
  --sql-target sql_query \
  --sql-template-id no_sql_template \
  --sql-template-topic label
```

## confirm collection data

example

```
python query_collection_info.py \
  --use-weaviate-cloud \
  --weaviate-collection-name Collection_b199b2c66f664e80b6d7b3111c1cd14c_packative \
  --weaviate-cluster-url https://qbr49scyqeeitim9am2xeq.c0.asia-southeast1.gcp.weaviate.cloud
```