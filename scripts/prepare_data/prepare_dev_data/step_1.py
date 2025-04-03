import concurrent.futures
import json
import os
from threading import Lock

from dotenv import load_dotenv
from text2sql.engine.generation import AzureGenerator, GCPGenerator
from text2sql.engine.generation.postprocessing import extract_first_code_block
from tqdm import tqdm

from utils.formatter import SelectorFormatterWExamples

INFERENCE_FOLDER = "inferences_v3"
GCP_API_KEY = ""


def load_fewshot_sample(file, desc_types, train_descs_folder):
    with open(file, "r") as f:
        sample = json.load(f)

    answer_text = json.dumps(sample["answer"], indent=2)
    sample["schema_descriptions"] = {}
    for desc_type in desc_types:
        desc_file = os.path.join(
            train_descs_folder, desc_type, f"{sample['db_id']}.txt"
        )
        with open(desc_file, "r") as f:
            desc = f.read()
        sample["schema_descriptions"][desc_type] = desc

    sample["answer_text"] = answer_text
    return sample


def load_schema_desc_map(data, desc_types, desc_folder):
    db_ids = {row["db_id"] for row in data}
    desc_map = {}
    for desc_type in desc_types:
        desc_map[desc_type] = {}
        for db_id in db_ids:
            desc_file = os.path.join(desc_folder, desc_type, f"{db_id}.txt")
            with open(desc_file, "r") as f:
                desc = f.read()
            desc_map[desc_type][db_id] = desc
    return desc_map


def process_datum(
    datum,
    desc_type,
    schema_desc_map,
    selector_formatter,
    generator,
    generator_config,
    dev_table_mapping,
):
    """Function to process a single datum that can be called by threads"""
    try:
        question, evidence = datum["nl_en_query"], datum["evidence"]
        schema_desc = schema_desc_map[desc_type][datum["db_id"]]

        messages = selector_formatter.generate_messages(schema_desc, question, evidence)

        generation = generator.generate(messages, **generator_config)
        gold_table_map = dev_table_mapping[datum["question_id"]]["table_map"]

        inference = {
            "db_id": datum["db_id"],
            "question_id": datum["question_id"],
            "sql_query": datum["sql_query"],
            "nl_en_query": question,
            "evidence": evidence,
            "messages": messages,
            "generation": generation,
            "parsed_generation": json.loads(extract_first_code_block(generation)),
            "gold_table_map": gold_table_map,
        }
        return inference
    except Exception as e:
        print(generation)
        print(f"Error processing datum {datum.get('question_id', 'unknown')}: {e}")
        return None


def main():
    load_dotenv()

    debug = False
    debug_limit = 20
    model = "gena-4o-2024-08-06"
    dev_file = "./data/dev.json"
    dev_table_mapping_file = "./data/dev_table_mapping.json"

    few_shot_sample_file_1 = "./sample_data/fewshot_sample_1.json"
    few_shot_sample_file_2 = "./sample_data/fewshot_sample_2.json"
    few_shot_sample_file_3 = "./sample_data/fewshot_sample_3.json"
    few_shot_sample_files = [few_shot_sample_file_1, few_shot_sample_file_2, few_shot_sample_file_3]
    

    desc_types = [
        # "dail_types_relations",
        # "m_schema",
        "mac_schema",
        # "json_raw",
        # "sql_create",
        # "sqlalchemy_tables",
    ]
    desc_folder = "./data/dev_descriptions"
    desc_train_folder = "./data/train_descriptions"
    max_workers = 4 

    with open(dev_file, "r") as f:
        dev_data = json.load(f)
    with open(dev_table_mapping_file, "r") as f:
        dev_table_mapping = json.load(f)
    dev_table_mapping = {row["question_id"]: row for row in dev_table_mapping}
    if debug:
        dev_data = dev_data[:debug_limit]

    few_shot_samples = [load_fewshot_sample(
        few_shot_sample_file, desc_types, desc_train_folder
    ) for few_shot_sample_file in few_shot_sample_files]


    model = "gemini-1.5-pro"
    generator = GCPGenerator(
        api_key=GCP_API_KEY,
        model=model,
    )

    schema_desc_map = load_schema_desc_map(dev_data, desc_types, desc_folder)

    generator_config = {"temperature": 0}
    for desc_type in desc_types:
        selector_formatter = SelectorFormatterWExamples(
            few_shot_samples,
            desc_type
        )

        inferences = []

        # Create a progress bar that will update across threads
        progress_bar = tqdm(total=len(dev_data), desc=f"Processing {desc_type}")
        progress_lock = Lock()

        # Function to update the progress bar safely from multiple threads
        def update_progress(_):
            with progress_lock:
                progress_bar.update(1)

        # Create a thread pool and submit tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_datum = {
                executor.submit(
                    process_datum,
                    datum,
                    desc_type,
                    schema_desc_map,
                    selector_formatter,
                    generator,
                    generator_config,
                    dev_table_mapping,
                ): datum
                for datum in dev_data
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_datum):
                result = future.result()
                if result:
                    inferences.append(result)
                update_progress(None)

        progress_bar.close()

        with open(f"./{INFERENCE_FOLDER}/{desc_type}_hr3s_gemini_fulldev.json", "w") as f:
            json.dump(inferences, f, indent=2)


if __name__ == "__main__":
    main()
