from abc import ABC, abstractmethod


class BasePromptFormatter(ABC):

    @abstractmethod
    def generate_messages(self, query: str) -> list[dict]:
        pass


class BasicFewShotPromptFormatter(BasePromptFormatter):

    def __init__(
            self,
            few_shot_query_key: str = "question",
            few_shot_target_key: str = "SQL",
            markdown_format: bool = False
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
