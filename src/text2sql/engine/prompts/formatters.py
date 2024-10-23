from abc import ABC, abstractmethod


class BasePromptFormatter(ABC):

    @abstractmethod
    def generate_messages(self, query: str) -> list[dict]:
        pass


class BasicFewShotPromptFormatter(BasePromptFormatter):

    def generate_messages(
            self, 
            system_message: str, 
            schema_description: str, 
            query: str, 
            few_shot_examples: list[dict],
            few_shot_query_key: str = "question",
            few_shot_target_key: str = "SQL"
        ) -> list[dict]:
        messages = [{"role": "system", "content": system_message}]
        for example in few_shot_examples:
            example_query = example["data"][few_shot_query_key]
            example_sql = example["data"][few_shot_target_key]
            messages.append({"role": "user", "content": f"similar example query: {example_query}"})
            messages.append({"role": "assistant", "content": example_sql})
        messages.append({"role": "user", "content": f"target schema:\n{schema_description}\n\nquery: {query}"})
        return messages
