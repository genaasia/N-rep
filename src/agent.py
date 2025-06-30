import json
import os

from models import AgenticRewriteResult, CandidateSelection
from text2sql.data import BaseDataset, SchemaManager
from text2sql.engine.generation import GCPGenerator
from text2sql.engine.generation.postprocessing import extract_first_code_block

TEMPERATURE = 0
MESSAGES_DIR = "./agentic_messages"  # Helps debug, will be removed before submission
MAX_RETRIES = 2
MAX_CORRECTIONS = 1  # Token count can be a problem with more corrections


class AgenticRewrite:
    def __init__(self, schema_manager: SchemaManager, dataset: BaseDataset):
        self.sys_p, self.u_p1_temp, self.u_p2_temp, self.u_p3_temp, self.u_p4_temp = self.load_agentic_prompts()
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
        idx = candidate.question_id
        question = candidate.question
        db_id = candidate.db_id

        schema = self.schema_manager.get_full_schema(db_id, "m_schema")

        u_p1 = self.u_p1_temp.replace("{SCHEMA}", schema).replace("{QUESTION}", question)
        chat_history = [
            {"role": "system", "content": self.sys_p},
            {"role": "user", "content": u_p1},
        ]
        response_json = None
        ass_response_1 = None
        for attempt in range(MAX_RETRIES):
            response = self.generator.generate(chat_history, temperature=TEMPERATURE)
            ass_response_1 = response.text
            try:
                response_json = self.parse_response(response.text)
                break
            except Exception as e:
                print(
                    f"Attempt {attempt+1} failed to parse LLM output. Retrying...\nError: {e}\nOutput: {response.text}"
                )
                continue
        if response_json is None:
            # Could not parse after all attempts, skip this question
            # Optionally, save chat history for debugging
            if ass_response_1:
                chat_history = chat_history + [{"role": "assistent", "content": ass_response_1}]

            self.save_chat_history(chat_history, idx, failed=True)

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
        u_p2 = self.u_p2_temp.replace("{QUESTION}", question).replace("{RESULTS}", json.dumps(response_json, indent=2))
        chat_history = [
            {"role": "system", "content": self.sys_p},
            {"role": "user", "content": u_p1},
            {"role": "assistent", "content": ass_response_1},
            {"role": "user", "content": u_p2},
        ]
        response_2 = self.generator.generate(chat_history, temperature=TEMPERATURE)
        final_sql_query = response_2.text.strip()
        final_sql_query_tmp = final_sql_query
        try:
            if "```" in final_sql_query:
                final_sql_query = extract_first_code_block(final_sql_query)
        except Exception as e:
            final_sql_query = final_sql_query_tmp
        final_query_result = self.dataset.validate_query(db_id, final_sql_query, timeout_secs=25)

        # Correction logic using prompt 3 and 4 if needed
        correction_count = 0
        while len(final_query_result["execution_result"]) == 0 and correction_count < MAX_CORRECTIONS:
            correction_count += 1
            final_sql_query, final_query_result, chat_history = self.do_correction(
                final_query_result, chat_history, question, db_id
            )

        self.save_chat_history(chat_history, idx)

        return AgenticRewriteResult(
            question_id=idx,
            question=question,
            db_id=db_id,
            rewritten_sql=final_sql_query,
        )

    def do_correction(self, final_query_result, chat_history, question, db_id):
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

        chat_history = chat_history + [{"role": "user", "content": u_p3}]
        response_3 = self.generator.generate(chat_history, temperature=TEMPERATURE)
        ass_response_3 = response_3.text
        # Extract exploratory queries (JSON array)
        exploratory_queries = []
        try:
            exploratory_queries = self.parse_response(ass_response_3)
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
        chat_history = chat_history + [
            {"role": "assistent", "content": ass_response_3},
            {"role": "user", "content": u_p4},
        ]
        response_4 = self.generator.generate(chat_history, temperature=TEMPERATURE)
        corrected_sql_query = response_4.text.strip()
        corrected_sql_query_tmp = corrected_sql_query
        try:
            if "```" in corrected_sql_query:
                corrected_sql_query = extract_first_code_block(corrected_sql_query)
        except Exception as e:
            corrected_sql_query = corrected_sql_query_tmp
        corrected_query_result = self.dataset.validate_query(db_id, corrected_sql_query, timeout_secs=25)
        # Use the correction
        final_sql_query = corrected_sql_query
        chat_history = chat_history + [
            {"role": "assistent", "content": response_4.text},
        ]

        return final_sql_query, corrected_query_result, chat_history

    def parse_response(self, response_text):
        if "```json" in response_text:
            return json.loads(extract_first_code_block(response_text))
        # delete everything before [ and after ]
        return json.loads(
            response_text.strip("")[response_text.strip().index("[") : response_text.strip().rindex("]") + 1]
        )

    def save_chat_history(self, chat_history, idx, failed=False):
        # Save chat history for this question
        chat_dir = os.path.join(MESSAGES_DIR, self.model_name)
        os.makedirs(chat_dir, exist_ok=True)
        file_name = f"question_{idx}" if not failed else f"question_{idx}_FAILED"
        chat_path = os.path.join(chat_dir, f"{file_name}.json")
        with open(chat_path, "w") as chat_f:
            json.dump(chat_history, chat_f, indent=2)
        chat_txt_path = os.path.join(chat_dir, f"{file_name}.txt")
        with open(chat_txt_path, "w") as txt_f:
            for msg in chat_history:
                txt_f.write(f"{msg['role'].upper()}:\n{msg['content']}\n\n")
