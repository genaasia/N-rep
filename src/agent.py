import json
import os

from text2sql.engine.generation import GCPGenerator
from text2sql.data import SchemaManager, BaseDataset

from text2sql.engine.generation.postprocessing import extract_first_code_block
from models import CandidateSelection, AgenticRewriteResult


TEMPERATURE = 0.1
MESSAGES_DIR = "./agentic_messages"


class AgenticRewrite:
    def __init__(self, schema_manager: SchemaManager, dataset: BaseDataset):
        self.sys_p, self.u_p1_temp, self.u_p2_temp, self.u_p3_temp, self.u_p4_temp = (
            self.load_agentic_prompts()
        )
        self.model_name = "gemini-2.5-flash"
        self.generator = GCPGenerator(
            model=self.model_name,
            api_key=os.getenv("GCP_KEY"),
        )
        self.schema_manager: SchemaManager = schema_manager
        self.dataset: BaseDataset = dataset

    def load_agentic_prompts(self):
        with open("./agentic_prompts/system_prompt.txt", "r") as f:
            sys_p = f.read()

        with open("./agentic_prompts/user_prompt_1.txt", "r") as f:
            u_p1_temp = f.read()

        with open("./agentic_prompts/user_prompt_2.txt", "r") as f:
            u_p2_temp = f.read()

        with open("./agentic_prompts/user_prompt_3.txt", "r") as f:
            u_p3_temp = f.read()

        with open("./agentic_prompts/user_prompt_4.txt", "r") as f:
            u_p4_temp = f.read()

        return sys_p, u_p1_temp, u_p2_temp, u_p3_temp, u_p4_temp

    def process_row(self, candidate: CandidateSelection):
        # idx, row, db_schema_cache, dataset, sys_p, u_p1_temp, u_p2_temp, model_name = args

        idx = candidate.question_id
        question = candidate.question  # GET QUESTION
        db_id = candidate.db_id

        row = {}

        schema = self.schema_manager.get_full_schema(db_id, "m_schema")

        u_p1 = self.u_p1_temp.replace("{SCHEMA}", schema).replace(
            "{QUESTION}", question
        )
        test_messages = [
            {"role": "system", "content": self.sys_p},
            {"role": "user", "content": u_p1},
        ]
        max_retries = 2
        response_json = None
        ass_response_1 = None
        for attempt in range(max_retries):
            response = self.generator.generate(test_messages, temperature=TEMPERATURE)
            ass_response_1 = response.text
            try:
                if "```json" in response.text:
                    response_json = json.loads(extract_first_code_block(response.text))
                    break
                else:
                    response_json = json.loads(
                        response.text.strip("")[
                            response.text.strip()
                            .index("[") : response.text.strip()
                            .rindex("]")
                            + 1
                        ]
                    )
                    break
            except Exception as e:
                print(
                    f"Attempt {attempt+1} failed to parse LLM output. Retrying...\nError: {e}\nOutput: {response.text}"
                )
                continue
        if response_json is None:
            # Could not parse after all attempts, skip this question
            row["prediction"] = "PARSING_FAILED"
            row["prediction_execution_result"] = []
            row["prediction_num_rows"] = 0
            # Optionally, save chat history for debugging
            chat_dir = os.path.join(MESSAGES_DIR, self.model_name)
            os.makedirs(chat_dir, exist_ok=True)
            chat_path = os.path.join(chat_dir, f"question_{idx}_FAILED.json")
            with open(chat_path, "w") as chat_f:
                json.dump(
                    {"messages": test_messages, "response": ass_response_1},
                    chat_f,
                    indent=2,
                )
            chat_txt_path = os.path.join(chat_dir, f"question_{idx}_FAILED.txt")
            with open(chat_txt_path, "w") as txt_f:
                for msg in test_messages:
                    txt_f.write(f"{msg['role'].upper()}:\n{msg['content']}\n\n")
                if ass_response_1:
                    txt_f.write(f"ASSISTENT:\n{ass_response_1}\n\n")
            return AgenticRewriteResult(
                question_id=idx,
                question=question,
                db_id=db_id,
                rewritten_sql="PARSING_FAILED",
            )
        for gq in response_json:
            query = gq["query"]
            result = self.dataset.validate_query(db_id, query, timeout_secs=5)
            if result.get("validated"):
                gq["db_result"] = result
            else:
                gq["db_result"] = result["message"]
        u_p2 = self.u_p2_temp.replace("{QUESTION}", question).replace(
            "{RESULTS}", json.dumps(response_json, indent=2)
        )
        test_messages_2 = [
            {"role": "system", "content": self.sys_p},
            {"role": "user", "content": u_p1},
            {"role": "assistent", "content": ass_response_1},
            {"role": "user", "content": u_p2},
        ]
        response_2 = self.generator.generate(test_messages_2, temperature=TEMPERATURE)
        final_sql_query = response_2.text.strip()
        final_sql_query_tmp = final_sql_query
        try:
            if "```" in final_sql_query:
                final_sql_query = extract_first_code_block(final_sql_query)
        except Exception as e:
            final_sql_query = final_sql_query_tmp
        final_query_result = self.dataset.validate_query(
            db_id, final_sql_query, timeout_secs=25
        )

        # needs_correction = len(final_query_result["execution_result"]) == 0

        # Correction logic using prompt 3 and 4 if needed
        correction_attempted = False
        max_corrections = 1
        correction_count = 0
        while (
            len(final_query_result["execution_result"]) == 0
            and correction_count < max_corrections
        ):
            correction_attempted = True
            correction_count += 1
            # --- PROMPT 3 ---
            # u_p3 = u_p3_temp.replace("{SCHEMA}", schema).replace("{QUESTION}", question)
            u_p3 = self.u_p3_temp.replace("{QUESTION}", question)

            if not final_query_result.get("validated"):
                error_message = (
                    "The final SQL query you wrote failed to execute. This is the error message:\n"
                    + final_query_result["message"]
                )
                u_p3 = u_p3.replace(
                    "The final SQL query you wrote returned an empty result.",
                    error_message,
                )

            test_messages_3 = test_messages_2 + [{"role": "user", "content": u_p3}]
            response_3 = self.generator.generate(
                test_messages_3, temperature=TEMPERATURE
            )
            ass_response_3 = response_3.text
            # Extract exploratory queries (JSON array)
            exploratory_queries = []
            try:
                if "```json" in ass_response_3:
                    exploratory_queries = json.loads(
                        extract_first_code_block(ass_response_3)
                    )
                else:
                    # delete everything before [ and after ]
                    exploratory_queries = json.loads(
                        ass_response_3.strip("")[
                            ass_response_3.strip()
                            .index("[") : ass_response_3.strip()
                            .rindex("]")
                            + 1
                        ]
                    )
            except Exception as e:
                exploratory_queries = []
            # Run exploratory queries and collect results
            exploratory_results = []
            for gq in exploratory_queries:
                query = gq.get("query") if isinstance(gq, dict) else None
                if query:
                    result = self.dataset.validate_query(db_id, query, timeout_secs=7)
                    gq_result = gq.copy() if isinstance(gq, dict) else {"query": query}
                    gq_result["db_result"] = result
                    exploratory_results.append(gq_result)
            # --- PROMPT 4 ---
            u_p4 = self.u_p4_temp.replace("{QUESTION}", question).replace(
                "{RESULTS}", json.dumps(exploratory_results, indent=2)
            )
            test_messages_4 = test_messages_3 + [
                {"role": "assistent", "content": ass_response_3},
                {"role": "user", "content": u_p4},
            ]
            response_4 = self.generator.generate(
                test_messages_4, temperature=TEMPERATURE
            )
            corrected_sql_query = response_4.text.strip()
            corrected_sql_query_tmp = corrected_sql_query
            try:
                if "```" in corrected_sql_query:
                    corrected_sql_query = extract_first_code_block(corrected_sql_query)
            except Exception as e:
                corrected_sql_query = corrected_sql_query_tmp
            corrected_query_result = self.dataset.validate_query(
                db_id, corrected_sql_query, timeout_secs=25
            )
            # Use the correction
            final_sql_query = corrected_sql_query
            final_query_result = corrected_query_result
            test_messages_2 = test_messages_4

        # original_sql = row.get("SQL", "").strip()
        # original_query_result = dataset.validate_query(db_id, original_sql, timeout_secs=25)
        # exec_match = execution_match(
        # final_query_result["execution_result"],
        # original_query_result["execution_result"],
        # )

        # Save only the first 10 rows for execution results, but if more than 50 rows, truncate and add a message
        pred_exec_result = final_query_result["execution_result"][:10]
        # real_exec_result = truncate_result(original_query_result["execution_result"])
        pred_num_rows = (
            len(final_query_result["execution_result"])
            if isinstance(final_query_result["execution_result"], list)
            else 0
        )
        # real_num_rows = (
        #     len(original_query_result["execution_result"])
        #     if isinstance(original_query_result["execution_result"], list)
        #     else 0
        # )
        row["prediction"] = final_sql_query
        row["prediction_execution_result"] = pred_exec_result
        # row["real_execution_result"] = real_exec_result
        row["prediction_num_rows"] = pred_num_rows
        # row["real_num_rows"] = real_num_rows
        # row["execution_match"] = exec_match
        row["correction_attempted"] = correction_attempted
        # Save chat history for this question
        chat_dir = os.path.join(MESSAGES_DIR, self.model_name)
        os.makedirs(chat_dir, exist_ok=True)
        chat_path = os.path.join(chat_dir, f"question_{idx}.json")
        with open(chat_path, "w") as chat_f:
            json.dump(test_messages_2, chat_f, indent=2)
        chat_txt_path = os.path.join(chat_dir, f"question_{idx}.txt")
        with open(chat_txt_path, "w") as txt_f:
            for msg in test_messages_2:
                txt_f.write(f"{msg['role'].upper()}:\n{msg['content']}\n\n")
        
        return AgenticRewriteResult(
            question_id=idx,
            question=question,
            db_id=db_id,
            rewritten_sql=final_sql_query,
        )
