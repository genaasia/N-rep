from .constants import SELECT_SYSTEM_PROMPT_V4, SELECTOR_EXAMPLE_PROMPT_V4, SELECTOR_USER_PROMPT_V4
class SelectorFormatterWExamples:
    """format messages in the GENA AI API format with custom prompt template."""

    def __init__(
        self,
        few_shot_examples : list[dict],
        desc_type:str,
    ):
        self.system_prompt = SELECT_SYSTEM_PROMPT_V4
        for example in few_shot_examples:
            example_description: str = example["schema_descriptions"][desc_type]
            example_question: str = example["question"]
            example_evidence: str = example["evidence"]
            example_answer: str = example["answer"]
            self.system_prompt += SELECTOR_EXAMPLE_PROMPT_V4.format(
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

        query_message = SELECTOR_USER_PROMPT_V4.format(
            schema_description=schema_description,
            question=question,
            evidence=evidence,
        )

        messages.append({"role": "user", "content": query_message})
        return messages
