import datetime

from abc import ABC, abstractmethod

from text2sql.engine.prompts.constants import (
    DEFAULT_FEWSHOT_SYSTEM_PROMPT,
    FEWSHOT_USER_EXAMPLE_TEMPLATE,
    FEWSHOT_USER_QUERY_TEMPLATE,
    FEWSHOT_ASSISTANT_TEMPLATE,
)

from text2sql.engine.prompts.constants_v2 import (
    CHESS_COT_PROMPT,
    ESQL_COT_PROMPT,
    ESQL_QE_PROMPT
)

from text2sql.engine.prompts.constants_v3 import SIMPLE_PROMPT_TEMPLATE_FOR_REASONERS, SIMPLE_USER_PROMPT_FOR_REASONERS
from text2sql.engine.prompts.constants_v3 import GENA_COT_PROMPT_TEMPLATE as GENA_COT_PROMPT_V3_TEMPLATE
from text2sql.engine.prompts.constants_v3 import GENA_REPAIR_SYSTEM_PROMPT_TEMPLATE
from text2sql.engine.prompts.constants_v3 import GENA_REPAIR_USER_MESSAGE_TEMPLATE
from text2sql.engine.prompts.constants_v3 import GENA_REWRITE_SYSTEM_PROMPT_TEMPLATE
from text2sql.engine.prompts.constants_v3 import GENA_REWRITE_USER_MESSAGE_TEMPLATE
from text2sql.engine.prompts.constants_v3 import (
     GENA_MYSQL_GUIDELINES,
     GENA_POSTGRES_GUIDELINES,
     GENA_SQLITE_GUIDELINES,
     GENA_USER_EXAMPLE_TEMPLATE,
     GENA_USER_QUERY_TEMPLATE,
     GENA_USER_QUERY_NO_DATE_TEMPLATE,
     GENA_USER_QUERY_EVIDENCE_SCHEMA_TEMPLATE,
     GENA_COT_W_EVIDENCE_PROMPT_TEMPLATE,
     GENA_ASSISTANT_TEMPLATE,
     REWRITE_PROMPT_TEMPLATE,
     REWRITE_USER_MESSAGE_TEMPLATE,
     GENA_USER_QUERY_SCHEMA_TEMPLATE
)


class BasePromptFormatter(ABC):

    @abstractmethod
    def generate_messages(self, query: str) -> list[dict]:
        pass


class BasicFewShotPromptFormatter(BasePromptFormatter):
    """basic formatter for few-shot prompted inference.
    
    this formatter generates a list of messages for few-shot prompted inference.
    this is configured for the "general" case, in which the schema information
    is only provided in the user message, and assumes the few-shot examples may
    be from other databases (e.g. BIRD, SPIDER cases).
    
    """
    def __init__(
            self,
            few_shot_query_key: str = "question",
            few_shot_target_key: str = "SQL",
            markdown_format: bool = True
        ):
            self.few_shot_query_key = few_shot_query_key
            self.few_shot_target_key = few_shot_target_key
            self.markdown_format = markdown_format


    def generate_messages(
            self, 
            system_message: str, 
            schema_description: str, 
            query: str, 
            few_shot_examples: list[dict] = [],
        ) -> list[dict]:
        messages = [{"role": "system", "content": system_message}]
        for example in few_shot_examples:
            example_query = example["data"][self.few_shot_query_key]
            example_sql = example["data"][self.few_shot_target_key]
            messages.append({"role": "user", "content": f"similar example query: {example_query}"})
            if self.markdown_format:
                output = f"```sql\n{example_sql}\n```"
            else:
                output = example_sql
            messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": f"target schema:\n{schema_description}\n\ntarget query: {query}"})
        return messages
    

class LegacyFewShotPromptFormatter(BasePromptFormatter):
    """prompt formatter for few-shot prompted inference based in legacy text2sql API.
    
    Like the api, the schema description is added to the system prompt.
    There is a soft assumption that the few-shot examples are from the same database.
    """
    def __init__(
            self,
            database_type: str,
            few_shot_query_key: str = "question",
            few_shot_target_key: str = "SQL"
        ):
            self.database_type = database_type
            self.few_shot_query_key = few_shot_query_key
            self.few_shot_target_key = few_shot_target_key

    def format_user_message(self, user_message: str, db_type: str, add_date: bool) -> str:
        """format a single message"""
        if add_date:
            current_date = datetime.datetime.now().strftime('%A, %B %d, %Y')  # like "Friday, November 1, 2024"
            return FEWSHOT_USER_QUERY_TEMPLATE.format(current_date=current_date, user_message=user_message, db_type=db_type)
        return FEWSHOT_USER_EXAMPLE_TEMPLATE.format(user_message=user_message, db_type=db_type)

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
            few_shot_examples: list[dict] = [],
            system_message: str | None = None, 
        ) -> list[dict]:
        # if system message is provided, use it and add schema description
        if system_message:
            if '{schema_description}' in system_message:
                 formatted_system_message = system_message.format(schema_description=schema_description)
            elif schema_description:
                formatted_system_message = f"{system_message}\n\ndatabase schema description:\n\n{schema_description}"
            messages = [{"role": "system", "content": formatted_system_message}]
        # use default system message by default
        else:
            formatted_system_message = DEFAULT_FEWSHOT_SYSTEM_PROMPT.format(schema_description=schema_description)
            messages = [{"role": "system", "content": formatted_system_message}]
        # assemble list of messages with few-shot examples
        for example in few_shot_examples:
            example_query = example["data"][self.few_shot_query_key]
            example_sql = example["data"][self.few_shot_target_key]
            example_message = self.format_user_message(example_query, self.database_type, add_date=False)
            messages.append({"role": "user", "content": example_message})
            output_message = FEWSHOT_ASSISTANT_TEMPLATE.format(sql_query=example_sql)
            messages.append({"role": "assistant", "content": output_message})
        # add date to the target query (for relative dates)
        query_message = self.format_user_message(query, self.database_type, add_date=True)
        messages.append({"role": "user", "content": query_message})
        return messages


class ChessCoTPromptFormatter(BasePromptFormatter):
    def __init__(
            self,
            database_type: str,
        ):
            self.database_type = database_type

    def format_user_message(self, schema_description: str, question: str) -> str:
        """format a single message"""
        return CHESS_COT_PROMPT.format(schema_description=schema_description, question=question)

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
        ) -> list[dict]:
        messages = []
        query_message = self.format_user_message(schema_description, query)
        messages.append({"role": "user", "content": query_message})
        return messages


class ESQLCoTPromptFormatter(BasePromptFormatter):
    def __init__(
            self,
            database_type: str,
        ):
            self.database_type = database_type

    def format_user_message(self, schema_description: str, question: str) -> str:
        """format a single message"""
        return ESQL_COT_PROMPT.format(schema_description=schema_description, question=question)

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
        ) -> list[dict]:
        messages = []
        query_message = self.format_user_message(schema_description, query)
        messages.append({"role": "user", "content": query_message})
        return messages


class ESQLQEPromptFormatter(BasePromptFormatter):
    def __init__(
            self,
            database_type: str,
        ):
            self.database_type = database_type

    def format_user_message(self, schema_description: str, question: str) -> str:
        """format a single message"""
        return ESQL_QE_PROMPT.format(schema_description=schema_description, question=question)

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
        ) -> list[dict]:
        messages = []
        query_message = self.format_user_message(schema_description, query)
        messages.append({"role": "user", "content": query_message})
        return messages


class GenaCoTPromptFormatter(BasePromptFormatter):
    """format messages in the GENA AI API format with custom prompt template."""
    def __init__(
            self,
            database_type: str,
            few_shot_query_key: str = "nl_en_query",
            few_shot_target_key: str = "sql_query",
            current_date: str = datetime.datetime.now().strftime("%A, %B %d, %Y"),
            add_date: bool = True,
        ):
        self.database_type = database_type
        self.few_shot_query_key = few_shot_query_key
        self.few_shot_target_key = few_shot_target_key
        self.current_date = current_date
        self.add_date = add_date

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
            few_shot_examples: list[dict] = [],
        ) -> list[dict]:
        if self.database_type == "mysql":
            dialect_guidelines = GENA_MYSQL_GUIDELINES
        elif self.database_type == "postgres":
            dialect_guidelines = GENA_POSTGRES_GUIDELINES
        elif self.database_type == "sqlite":
            dialect_guidelines = GENA_SQLITE_GUIDELINES
        else:
            raise ValueError(f"unsupported database type: {self.database_type}")

        formatted_system_message = GENA_COT_PROMPT_V3_TEMPLATE.format(
            sql_dialect=self.database_type,
            dialect_guidelines=dialect_guidelines,
            schema_description=schema_description
        )
        messages = [{"role": "system", "content": formatted_system_message}]

        for example in few_shot_examples:
            example_query = example["data"][self.few_shot_query_key]
            example_sql = example["data"][self.few_shot_target_key]
            messages.append({"role": "user", "content": GENA_USER_EXAMPLE_TEMPLATE.format(user_question=example_query, sql_dialect=self.database_type)})
            messages.append({"role": "assistant", "content": GENA_ASSISTANT_TEMPLATE.format(sql_query=example_sql)})

        if self.add_date:
            query_message = GENA_USER_QUERY_TEMPLATE.format(
                current_date=self.current_date, 
                user_question=query, 
                sql_dialect=self.database_type
            )
        else:
            query_message = GENA_USER_QUERY_NO_DATE_TEMPLATE.format(
                current_date=self.current_date, 
                user_question=query, 
                sql_dialect=self.database_type
            )

        messages.append({"role": "user", "content": query_message})
        return messages
    

class GenaRepairPromptFormatter(BasePromptFormatter):
    """format messages for repair with custom prompt template."""
    def __init__(
            self,
            database_type: str,
        ):
        self.database_type = database_type

    def generate_messages(
            self, 
            schema_description: str, 
            table_text: str,
            query: str, 
            predicted_sql: str,
            error: str | None = None,
        ) -> list[dict]:

        formatted_system_message = GENA_REPAIR_SYSTEM_PROMPT_TEMPLATE.format(
            sql_dialect=self.database_type
        )
        messages = [{"role": "system", "content": formatted_system_message}]

        query_message = GENA_REPAIR_USER_MESSAGE_TEMPLATE.format(
             sql_dialect=self.database_type,
             schema_description=schema_description,
             relevant_tables=table_text,
             user_question=query,
             original_sql=predicted_sql,
             error_message=error
        )
        messages.append({"role": "user", "content": query_message})
        return messages


class GenaRewritePromptFormatter(BasePromptFormatter):
    """format messages for rewrite with custom prompt template."""
    def __init__(
            self,
            database_type: str,
        ):
        self.database_type = database_type

    def generate_messages(
            self, 
            schema_description: str, 
            table_text: str,
            query: str, 
            predicted_sql: str,
        ) -> list[dict]:

        formatted_system_message = GENA_REWRITE_SYSTEM_PROMPT_TEMPLATE.format(
            sql_dialect=self.database_type
        )
        messages = [{"role": "system", "content": formatted_system_message}]

        query_message = GENA_REWRITE_USER_MESSAGE_TEMPLATE.format(
             sql_dialect=self.database_type,
             schema_description=schema_description,
             relevant_tables=table_text,
             user_question=query,
             original_sql=predicted_sql      
        )
        messages.append({"role": "user", "content": query_message})
        return messages


class SimplePromptFormatter(BasePromptFormatter):
    """format messages in a simple format for reasoning models."""
    def __init__(
            self,
            database_type: str,
        ):
        self.database_type = database_type

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
        ) -> list[dict]:
        if self.database_type == "mysql":
            dialect_guidelines = GENA_MYSQL_GUIDELINES
        elif self.database_type == "postgres":
            dialect_guidelines = GENA_POSTGRES_GUIDELINES
        elif self.database_type == "sqlite":
            dialect_guidelines = GENA_SQLITE_GUIDELINES
        else:
            raise ValueError(f"unsupported database type: {self.database_type}")

        formatted_system_message = SIMPLE_PROMPT_TEMPLATE_FOR_REASONERS.format(
            sql_dialect=self.database_type,
            dialect_guidelines=dialect_guidelines,
        )
        messages = [{"role": "system", "content": formatted_system_message}]

        query_message = SIMPLE_USER_PROMPT_FOR_REASONERS.format(
            schema_description=schema_description,
            user_question=query, 
            sql_dialect=self.database_type
        )

        messages.append({"role": "user", "content": query_message})
        return messages


class GenaCoTwEvidencePromptFormatter(BasePromptFormatter):
    """format messages in the GENA AI API format with custom prompt template."""
    def __init__(
            self,
            database_type: str,
            few_shot_query_key: str = "nl_en_query",
            few_shot_target_key: str = "sql_query",
            fewshot_schema_key: str | None = None,
        ):
        self.database_type = database_type
        self.few_shot_query_key = few_shot_query_key
        self.few_shot_target_key = few_shot_target_key
        self.fewshot_schema_key = fewshot_schema_key

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
            evidence: str,
            few_shot_examples: list[dict] = [],
        ) -> list[dict]:
        if self.database_type == "mysql":
            dialect_guidelines = GENA_MYSQL_GUIDELINES
        elif self.database_type == "postgres":
            dialect_guidelines = GENA_POSTGRES_GUIDELINES
        elif self.database_type == "sqlite":
            dialect_guidelines = GENA_SQLITE_GUIDELINES
        else:
            raise ValueError(f"unsupported database type: {self.database_type}")

        formatted_system_message = GENA_COT_W_EVIDENCE_PROMPT_TEMPLATE.format(
            sql_dialect=self.database_type,
            dialect_guidelines=dialect_guidelines,
            # schema_description=schema_description
        )
        messages = [{"role": "system", "content": formatted_system_message}]

        if few_shot_examples and self.fewshot_schema_key is None:
            raise ValueError("fewshot_schema_key is not provided")

        for example in few_shot_examples:
            example_query = example["data"][self.few_shot_query_key]
            example_sql = example["data"][self.few_shot_target_key]
            example_desc = example["data"][self.fewshot_schema_key]
            if "evidence" in example["data"]:
                example_evidence = example["data"]["evidence"]
                messages.append({"role": "user", "content": GENA_USER_QUERY_EVIDENCE_SCHEMA_TEMPLATE.format(schema_description=example_desc, user_question=example_query, evidence=example_evidence, sql_dialect=self.database_type)})
            else:
                messages.append({"role": "user", "content": GENA_USER_QUERY_SCHEMA_TEMPLATE.format(schema_description=example_desc, user_question=example_query, sql_dialect=self.database_type)})
            messages.append({"role": "assistant", "content": GENA_ASSISTANT_TEMPLATE.format(sql_query=example_sql)})

        query_message = GENA_USER_QUERY_EVIDENCE_SCHEMA_TEMPLATE.format(
            schema_description=schema_description,
            user_question=query, 
            evidence=evidence, 
            sql_dialect=self.database_type
        )

        messages.append({"role": "user", "content": query_message})
        return messages


class RewritePromptFormatter(BasePromptFormatter):
    """format messages for rewrite with custom prompt template."""
    def __init__(
            self,
            database_type: str,
        ):
        self.database_type = database_type

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
            predicted_sql: str,
        ) -> list[dict]:

        formatted_system_message = REWRITE_PROMPT_TEMPLATE.format(
            sql_dialect=self.database_type
        )
        messages = [{"role": "system", "content": formatted_system_message}]

        query_message = REWRITE_USER_MESSAGE_TEMPLATE.format(
             schema_description=schema_description,
            #  relevant_tables=table_text,
             user_question=query,
             original_sql=predicted_sql      
        )
        messages.append({"role": "user", "content": query_message})
        return messages


from text2sql.engine.prompts.constants_schema_linking import (
    SCHEMA_LINKING_SYSTEM_PROMPT, 
    SCHEMA_LINKING_EXAMPLE_PROMPT_TEMPLATE,
    SCHEMA_LINKING_USER_PROMPT_TEMPLATE
)

class SelectorFormatterWExamples:
    """format messages in the GENA AI API format with custom prompt template."""

    def __init__(
        self,
        few_shot_examples : list[dict],
        desc_type:str,
    ):
        self.system_prompt = SCHEMA_LINKING_SYSTEM_PROMPT
        for example in few_shot_examples:
            example_description: str = example["schema_descriptions"][desc_type]
            example_question: str = example["question"]
            example_evidence: str = example["evidence"]
            example_answer: str = example["answer"]
            self.system_prompt += SCHEMA_LINKING_EXAMPLE_PROMPT_TEMPLATE.format(
                example_description=example_description,
                example_question=example_question,
                example_evidence=example_evidence,
                example_answer=example_answer,
            )
        
        print(self.system_prompt)

    def generate_messages(
        self,
        schema_description: str,
        question: str,
        evidence: str,
    ) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]

        query_message = SCHEMA_LINKING_USER_PROMPT_TEMPLATE.format(
            schema_description=schema_description,
            question=question,
            evidence=evidence,
        )

        messages.append({"role": "user", "content": query_message})
        return messages
