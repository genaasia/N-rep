PROMPT = """I have a food delivery database with the following schema:

{schema}

I need you to generate EXACTLY {{n}} realistic records in JSON format for the {table_name} table (where n is a number you determine based on your reasoning, but must be 50 or less).

Before generating the data, think through these considerations:
1. What would be a realistic number of records given the reference data?
2. What are typical industry patterns for this type of data?
3. How should the values relate to the reference data to maintain realism?

First, write your thinking process in <reasoning> tags. Consider:
- Realistic ratios and relationships between tables
- Timing patterns and date relationships
- Typical business metrics and industry standards
- How the data you generate will impact key business metrics
- The specific number of records you chose to generate and why

Then, generate the COMPLETE set of records where:
- Each record must be a dictionary/object
- Keys must exactly match the column names from the schema
- Data types must match the column definitions
- Values should be realistic for a food delivery platform
- For foreign keys, use ONLY IDs from the reference data
- For ENUM fields, use only values specified in the schema
- Timestamp/datetime fields should use ISO format: YYYY-MM-DD HH:MM:SS
- Generated IDs should be between 1 and 100

This table's structure is:
{spesific_table_definition}

IMPORTANT: Generate the complete dataset with no omissions. Do not truncate or abbreviate the output.

Respond in this format:
<reasoning>
Your chain of thought about how many records to generate and why...
What patterns or distributions you'll use...
How you'll maintain realistic relationships...
</reasoning>

```json
[
    // Complete dataset here
    // Do not use ellipsis or omit any records
]
```"""


def get_table_definition(schema_text: str, table_name: str) -> str:
    lines = schema_text.split("\n")
    for line in lines:
        if not line.strip() or line.startswith("Relations:"):
            continue
        if line.startswith(f"table '{table_name}'"):
            return line.strip()
    return None


def format_prompt(schema: str, table_name: str, fk_prompt) -> str:
    spesific_table_definition = get_table_definition(schema, table_name)
    prompt = PROMPT.format(
        schema=schema,
        table_name=table_name,
        spesific_table_definition=spesific_table_definition,
    )
    if fk_prompt:
        prompt += f"\n\n{fk_prompt}"
    return prompt
