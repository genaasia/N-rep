from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel

from text2sql.data.datasets import SCHEMA_FORMATS
from text2sql.engine.generation import GenerationResult, TokenUsage


def update_moving_average(current_avg, n, new_sample):
    return (current_avg * n + new_sample) / (n + 1)


def verify_schema_format(schema_format: str):
    if schema_format not in SCHEMA_FORMATS:
        raise ValueError(f"Invalid schema format: {schema_format}")
    return schema_format


class SchemaLinkingInfo(BaseModel):
    question_id: int
    model_name: str
    schema_format: Annotated[str, AfterValidator(verify_schema_format)]
    messages: list[dict]
    generator_output: GenerationResult
    prediction: str
    table_linking: dict | None
    column_linking: dict | None
    table_description: str
    column_description: str
    full_description: str


class RewriteInfo(BaseModel):
    question_id: int
    original_sql: str
    rewritten_sql: str
    is_rewritten: bool  # whether the candidate was rewritten successfully (even if rewritten sql is same)
    messages: list[dict]
    generator_output: GenerationResult | None


class Candidate(BaseModel):
    question_id: int
    config_index: int
    sample: dict
    schema_format: Annotated[str, AfterValidator(verify_schema_format)]
    schema_filtering: Literal["none", "table", "column"]
    messages: list[dict]
    generator_output: GenerationResult
    original_sql: str  # first generation parsed result
    candidate_sql: str  # final candidate sql after rewrite
    rewrite_checked: bool = False
    rewrite_info: list[RewriteInfo] = []


class CandidateList(BaseModel):
    question_id: int
    candidate_configs: list[dict]
    candidates: list[Candidate]


class CandidateSelection(BaseModel):
    question_id: int
    db_id: str  # for formatting output
    generator_outputs: list[GenerationResult] = []
    candidate_config: dict
    selected_idx: int
    selected_sql: str
    max_vote_regular: int
    max_vote_chase: int
    needs_agentic_rewrite: bool
    question: str


class AgenticRewriteResult(BaseModel):
    question_id: int
    question: str
    db_id: str
    rewritten_sql: str


class TotalTokenUsage(BaseModel):
    label: str = ""
    calls: int = 0
    avg_inf_time_ms: float = 0
    tokens: TokenUsage = TokenUsage(
        prompt_tokens=0, output_tokens=0, total_tokens=0, inf_time_ms=0
    )

    # allow adding to TokenUsage. add to internal tokens TokenUsage and increment calls by one
    def __add__(self, other: TokenUsage) -> "TotalTokenUsage":
        self.tokens += other
        self.calls += 1
        self.avg_inf_time_ms = update_moving_average(
            self.avg_inf_time_ms, self.calls, other.inf_time_ms
        )
        return self


class TokenReport(BaseModel):
    total: TotalTokenUsage
    schema_linking: dict[str, TotalTokenUsage]
    sql_generation: TotalTokenUsage
    sql_generation_rewrite: TotalTokenUsage
    candidate_selection: TotalTokenUsage
    embedding: dict
