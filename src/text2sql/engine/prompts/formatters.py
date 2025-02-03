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
    ESQL_QE_PROMPT,
    GENA_COT_PROMPT,
    GENA_COT_PROMPT_ZERO,
    GENA_COT_USER_PROMPT
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
    def format_user_message(self, user_message) -> str:
        """format a single message"""
        return GENA_COT_USER_PROMPT.format(user_message=user_message)

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
            few_shot_examples: list[dict] = [],
        ) -> list[dict]:

        # if system message is provided, use it and add schema description
        formatted_system_message = GENA_COT_PROMPT.format(schema_description=schema_description)
        messages = [{"role": "system", "content": formatted_system_message}]

        for example in few_shot_examples:
            example_query = example["data"]["nl_en_query"]
            example_sql = example["data"]["sql_query"]
            messages.append({"role": "user", "content": f"text query: {example_query}"})
            output = f"```sql\n{example_sql}\n```"
            messages.append({"role": "assistant", "content": output})

        query_message = self.format_user_message(query)
        messages.append({"role": "user", "content": query_message})
        return messages

class GenaCoTZsPromptFormatter(BasePromptFormatter):
    def format_user_message(self, user_message) -> str:
        """format a single message"""
        return GENA_COT_USER_PROMPT.format(user_message=user_message)

    def generate_messages(
            self, 
            schema_description: str, 
            query: str, 
        ) -> list[dict]:

        # if system message is provided, use it and add schema description
        formatted_system_message = GENA_COT_PROMPT_ZERO.format(schema_description=schema_description)
        messages = [{"role": "system", "content": formatted_system_message}]

        query_message = self.format_user_message(query)
        messages.append({"role": "user", "content": query_message})
        return messages