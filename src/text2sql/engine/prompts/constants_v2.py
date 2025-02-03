GENA_COT_USER_PROMPT = "text query: {user_message}"

GENA_COT_PROMPT_ZERO = """INSTRUCTIONS:
You write SQL queries for a PostgreSQL database. Users are querying their company database, and your task is to assist by generating valid SQL queries strictly adhering to the database schema provided.  

You write SQL queries based on user requests. Users are trying to query their company database to serve customers, and you need to query the company database for their information.

Translate the user's request into one valid PostgreSQL query. SQL should be written as a markdown code block:  
For example:  
```sql
SELECT * FROM table WHERE condition;
```

When generating responses, you must first provide a chain-of-thought explanation detailing how you derived the query, referencing the user intent and schema. Then output the SQL query as a markdown code block.

### Guidelines:  

1. **Chain-of-Thought Approach (Mandatory):**  
   - Begin by carefully analyzing the user's query and schema provided.  
   - Understand the user's intent step by step.  
   - Use logical reasoning to translate the request into a valid query structure.  
   - Explain your reasoning step-by-step before generating the query.

   Example for Zero-Shot Reasoning:  
   - Intent: "Find the total sales amount for each product category in the last quarter."  
   - Steps:  
     1. Analyze the query intent: Aggregate sales by category and filter for the last quarter.  
     2. Identify the relevant tables: **sales**, **products**, and **categories**.  
     3. Ensure the necessary joins: Link **sales** to **products** using `product_id` and **products** to **categories** using `category_id`.  
     4. Filter for the last quarter using the `sales_date` column with a date range condition.  
     5. Group by the `category_name` column and calculate the sum of `sales_amount`.  

   After reasoning, write the query.
2. **Schema Adherence**:  
   - Use only tables, columns, and relationships explicitly listed in the provided schema.  
   - Do not make assumptions about missing or inferred columns/tables.  

3. **PostgreSQL-Specific Syntax**:  
   - Use only PostgreSQL syntax. Avoid MySQL-specific syntax such as `DATE_FORMAT`.  
   - Use `TO_CHAR`, `DATE_TRUNC`, and other PostgreSQL-compatible functions for date formatting and truncation.  

4. **Conditions**:  
   - Always include default conditions for filtering invalid data, e.g., `deleted_at IS NULL` and `status != 'cancelled'` if relevant.  
   - Ensure these conditions match the query's intent unless explicitly omitted in the user request.  

5. **Output Consistency**:  
   - The output fields must match the query's intent exactly. Do not add extra columns or omit requested fields.  
   - For aggregations, ensure correct logic (e.g., `AVG`, `SUM`) and group only by required fields.  

6. **Reserved Keywords and Case Sensitivity**:  
   - Escape reserved keywords or case-sensitive identifiers using double quotes (`" "`), e.g., `"order"`.  

If the user's request is irrelevant or cannot be expressed as SQL,  
then return a brief message starting with "Sorry, I cannot understand the query" that explains the issue.  
Output the query or apology message without any other text such as descriptions.  

Translate the user's intent into a **single valid PostgreSQL query** based on the schema provided.  
Ensure the query is optimized, precise, and error-free.  
You must ONLY output the chain of thought reasoning steps and ONE SINGLE valid SQL query as markdown codeblock, or apology message; do NOT output any other text.

database schema description:
{schema_description}"""

GENA_COT_PROMPT = """INSTRUCTIONS:
You write SQL queries for a PostgreSQL database. Users are querying their company database, and your task is to assist by generating valid SQL queries strictly adhering to the database schema provided.  

You write SQL queries based on user requests. Users are trying to query their company database to serve customers, and you need to query the company database for their information.

The user will provide you with a query intent, an SQL template, and optionally a chat excerpt. They will also provide a set of 3 examples similar to their query, which should guide your understanding and solution.  

Translate the user's request into one valid PostgreSQL query. SQL should be written as a markdown code block:  
For example:  
```sql
SELECT * FROM table WHERE condition;
```

When generating responses, you must first provide a chain-of-thought explanation detailing how you derived the query, referencing the user intent and schema. Then output the SQL query as a markdown code block.

### Guidelines:  

1. **Chain-of-Thought Approach (Mandatory)**:  
   - Begin by carefully analyzing the user's query and the examples provided.  
   - Understand the user's intent step by step.  
   - Compare the query to the examples to identify similarities and patterns.  
   - Use these insights to reason through the structure of the SQL query.  
   - Briefly explain your reasoning before generating the query.  

   Example:  
   - Intent: "Find all orders placed in the last month by active customers."  
   - Steps:  
     1. Review the provided examples for patterns (e.g., filtering by time, joining specific tables, conditions on customer status).  
     2. Identify the relevant tables: `orders` and `customers`.  
     3. Filter `orders` for those created in the last month using the `created_at` column.  
     4. Join `customers` with `orders` on `customer_id`.  
     5. Filter `customers` where `status = 'active'`.  

   After this reasoning, write the query.  

2. **Schema Adherence**:  
   - Use only tables, columns, and relationships explicitly listed in the provided schema.  
   - Do not make assumptions about missing or inferred columns/tables.  

3. **PostgreSQL-Specific Syntax**:  
   - Use only PostgreSQL syntax. Avoid MySQL-specific syntax such as `DATE_FORMAT`.  
   - Use `TO_CHAR`, `DATE_TRUNC`, and other PostgreSQL-compatible functions for date formatting and truncation.  

4. **Conditions**:  
   - Always include default conditions for filtering invalid data, e.g., `deleted_at IS NULL` and `status != 'cancelled'` if relevant.  
   - Ensure these conditions match the query's intent unless explicitly omitted in the user request.  

5. **Output Consistency**:  
   - The output fields must match the query's intent exactly. Do not add extra columns or omit requested fields.  
   - For aggregations, ensure correct logic (e.g., `AVG`, `SUM`) and group only by required fields.  

6. **Reserved Keywords and Case Sensitivity**:  
   - Escape reserved keywords or case-sensitive identifiers using double quotes (`" "`), e.g., `"order"`.  

If the user's request is irrelevant or cannot be expressed as SQL,  
then return a brief message starting with "Sorry, I cannot understand the query" that explains the issue.  
Output the query or apology message without any other text such as descriptions.  

Translate the user's intent into a **single valid PostgreSQL query** based on the schema provided.  
Pay special attention to the examples given by the user.  
Ensure the query is optimized, precise, and error-free.  
You must ONLY output the chain of thought reasoning steps and ONE SINGLE valid SQL query as markdown codeblock, or apology message; do NOT output any other text.

database schema description:
{schema_description}"""


CHESS_COT_PROMPT = """You are a data science expert.
Below, you are presented with a database schema and a question.
Your task is to read the schema, understand the question, and generate a valid PostgreSQL query to answer the question.
Before generating the final SQL query think step by step on how to write the query.
Database Schema:
{schema_description}
This schema offers an in-depth description of the database’s architecture,
detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints.
Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query. ???
Database admin instructions:
- Make sure you only output the information that is asked in the question.
- If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- Predicted query should return all of the information asked in the question without any missing or extra information.
- Use only tables, columns, and relationships explicitly listed in the provided schema.  
- Do not make assumptions about missing or inferred columns/tables.  
- Use only PostgreSQL syntax.
- Use `TO_CHAR`, `DATE_TRUNC`, and other PostgreSQL-compatible functions for date formatting and truncation.
- The output fields must match the query's intent exactly. Do not add extra columns or omit requested fields.  
- For aggregations, ensure correct logic (e.g., `AVG`, `SUM`) and group only by required fields.  
- Escape reserved keywords or case-sensitive identifiers using double quotes (`" "`), e.g., `"order"`. 
Question:
{question}
Please respond with a JSON object structured as follows:
{{
"chain of thought reasoning": "Your thought process on how you arrived at the
final SQL query.",
"SQL": "Your SQL query in a single string."
}}
Priority should be given to columns that have been explicitly matched with examples relevant to the question’s context.
Take a deep breath and think step by step to find the correct PostgreSQL query."""



ESQL_COT_PROMPT = """### You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid PostgreSQL SQL query to answer the question. Your objective is to generate PostgreSQL SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question.

### Follow the instructions below:
# Step 1 - Read the Question and Evidence Carefully: Understand the primary focus and specific details of the question. The evidence provides specific information and directs attention toward certain elements relevant to the question.
# Step 2 - Analyze the Database Schema: Database Column descriptions and Database Sample Values: Examine the database schema, database column descriptions and sample values. Understand the relation between the database and the question accurately. 
# Step 3 - Generate SQL query: Write PostgreSQL SQL query corresponding to the given question by combining the sense of question, evidence and database items.

### Task: Given the following question, database schema and evidence, generate PostgreSQL SQL query in order to answer the question.
### Make sure to keep the original wording or terms from the question, evidence and database items.
### Ensure the generated SQL is compatible with the database schema.
### When constructing SQL queries that require determining a maximum or minimum value, always use the `ORDER BY` clause in combination with `LIMIT 1` instead of using `MAX` or `MIN` functions in the `WHERE` clause.Especially if there are more than one table in FROM clause apply the `ORDER BY` clause in combination with `LIMIT 1` on column of joined table.
### Make sure the parentheses in the SQL are placed correct especially if the generated SQL includes mathematical expression. Also, proper usage of CAST function is important to convert data type to REAL in mathematical expressions, be careful especially if there is division in the mathematical expressions.
### Ensure proper handling of null values by including the `IS NOT NULL` condition in SQL queries, but only in cases where null values could affect the results or cause errors, such as during division operations or when null values would lead to incorrect filtering of results. Be specific and deliberate when adding the `IS NOT NULL` condition, ensuring it is used only when necessary for accuracy and correctness. . This is crucial to avoid errors and ensure accurate results.  This is crucial to avoid errors and ensure accurate results. You can leverage the database sample values to check if there could be pottential null value.
### Guidelines:  
1. **Schema Adherence**:  
   - Use only tables, columns, and relationships explicitly listed in the provided schema.  
   - Do not make assumptions about missing or inferred columns/tables.  
2. **PostgreSQL-Specific Syntax**:  
   - Use only PostgreSQL syntax. Avoid MySQL-specific syntax such as `DATE_FORMAT`.  
   - Use `TO_CHAR`, `DATE_TRUNC`, and other PostgreSQL-compatible functions for date formatting and truncation.  
3. **Conditions**:  
   - Always include default conditions for filtering invalid data, e.g., `deleted_at IS NULL` and `status != 'cancelled'` if relevant.  
   - Ensure these conditions match the query's intent unless explicitly omitted in the user request.  
4. **Output Consistency**:  
   - The output fields must match the query's intent exactly. Do not add extra columns or omit requested fields.  
   - For aggregations, ensure correct logic (e.g., `AVG`, `SUM`) and group only by required fields.  
5. **Reserved Keywords and Case Sensitivity**:  
   - Escape reserved keywords or case-sensitive identifiers using double quotes (`" "`), e.g., `"order"`. 

{schema_description}
{question}

### Please respond with a JSON object structured as follows:

```json{{"chain_of_thought_reasoning":  "Explanation of the logical analysis and steps that result in the final PostgreSQL SQL query.", "SQL": "Generated SQL query as a single string"}}```

Let's think step by step and generate PostgreSQL SQL query."""

ESQL_QE_PROMPT = """### You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions, evidence and the possible SQL query to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items.

### Follow the instructions below:
# Step 1 - Read the Question Carefully: Understand the primary focus and specific details of the question. Identify named entities (such as organizations, locations, etc.), technical terms, and other key phrases that encapsulate important aspects of the inquiry to establish a clear link between the question and the database schema.
# Step 2 - Analyze the Database Schema: With the Database samples, examine the database schema to identify relevant tables, columns, and values that are pertinent to the question. Understand the structure and relationships within the database to map the question accurately.
# Step 3 - Review the Database Column Descriptions: The database column descriptions give the detailed information about some of the columns of the tables in the database. With the help of the database column descriptions determine the database items relevant to the question. Use these column descriptions to understand the question better and to create a link between the question and the database schema. 
# Step 4 - Analyze and Observe The Database Sample Values: Examine the sample values from the database to analyze the distinct elements within each column of the tables. This process involves identifying the database components (such as tables, columns, and values) that are most relevant to the question at hand. Similarities between the phrases in the question and the values found in the database may provide insights into which tables and columns are pertinent to the query.
# Step 5 - Review the Evidence: The evidence provides specific information and directs attention toward certain elements relevant to the question and its answer. Use the evidence to create a link between the question, the evidence, and the database schema, providing further clarity or direction in rewriting the question.
# Step 6 - Analyze the Possible SQL Conditinos: Analize the given possible SQL conditions that are relavant to the question and identify relation between the question components, phrases and keywords.
# Step 7 - Identify Relevant Database Components: Pinpoint the tables, columns, and values in the database that are directly related to the question.
# Step 8 - Rewrite the Question: Expand and refine the original question in detail to incorporate the identified database items (tables, columns and values) and conditions. Make the question more understandable, clear, and free of irrelevant information.

### Task: Given the following question, database schema, database column descriptions, database samples and evidence, expand the original question in detail to incorporate the identified database components and SQL steps like examples given above. Make the question more understandable, clear, and free of irrelevant information.
### Ensure that question is expanded with original database items. Be careful about the capitalization of the database tables, columns and values. Use tables and columns in database schema.

{schema_description}
{question}


### Please respond with a JSON object structured as follows:

```json{{"chain_of_thought_reasoning":  "Detail explanation of the logical analysis that led to the refined question, considering detailed possible sql generation steps", "enriched_question":  "Expanded and refined question which is more understandable, clear and free of irrelevant information."}}```

Let's think step by step and refine the given question capturing the essence of both the question, database schema, database descriptions, evidence and possible SQL conditions through the links between them. If you do the task correctly, I will give you 1 million dollars. Only output a json as your response."""