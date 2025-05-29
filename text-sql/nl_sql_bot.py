import sqlite3
import re
import logging
import argparse
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Optional, Union
from tabulate import tabulate

# -------------------------------
# Logger setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------------
# Utility Functions

def sanitize_sql(sql: str) -> bool:
    """
    Basic sanity check for generated SQL.
    Returns True if SQL is considered safe for execution, else False.
    
    This is a very rudimentary check — real production systems should
    have full SQL parsing and validation or sandboxing.
    """
    sql_lower = sql.lower()
    forbidden_phrases = ["drop ", "delete ", "alter ", "update ", "insert ", "truncate ", "--"]
    for phrase in forbidden_phrases:
        if phrase in sql_lower:
            logger.warning(f"Potentially dangerous SQL phrase detected: {phrase.strip()}")
            return False
    return True

def pretty_print_result(rows: List[tuple], headers: Optional[List[str]] = None):
    if not rows:
        print("No rows returned.")
        return
    if not headers:
        # Fallback to generic headers
        headers = [f"col_{i}" for i in range(len(rows[0]))]
    print(tabulate(rows, headers=headers, tablefmt="psql"))

def extract_schema_sqlite(db_path: str) -> str:
    """
    Extract the schema from SQLite DB as a descriptive string to feed into prompt.
    This includes table names and columns.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema_desc = []
        for table_name_tuple in tables:
            table = table_name_tuple[0]
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
            col_desc = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
            schema_desc.append(f"Table '{table}': columns = {col_desc}")
        return "\n".join(schema_desc)
    except Exception as e:
        logger.error(f"Error extracting schema: {e}")
        return ""
    finally:
        conn.close()

# -------------------------------
# Model Handling Class

class NLToSQLModel:
    def __init__(self, model_name: str = "microsoft/phi-2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model {model_name} on device {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def generate_sql(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # The model may generate the prompt + the SQL, so strip prompt from output:
        sql_output = generated_text[len(prompt):].strip()
        # Further cleanup: often models output extra text after SQL, try to stop at first ";"
        if ";" in sql_output:
            sql_output = sql_output[:sql_output.index(";")+1]
        return sql_output

# -------------------------------
# DB Execution Class

class SQLExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute_sql(self, sql_query: str) -> Union[List[tuple], str]:
        if not sanitize_sql(sql_query):
            return "SQL query flagged as potentially unsafe. Execution aborted."
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            if sql_query.strip().lower().startswith("select"):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return rows, columns
            else:
                conn.commit()
                return "Query executed successfully."
        except Exception as e:
            return f"Error executing query: {e}"
        finally:
            conn.close()

# -------------------------------
# Prompt Engineering Helpers

def build_prompt(schema_description: str, user_request: str) -> str:
    """
    Build a prompt for the model that includes:
    - Instructions for the task
    - DB schema context
    - Few-shot examples (optional for demonstration)
    - User request
    """
    system_instructions = (
        "You are an AI assistant that generates valid SQLite SQL queries "
        "based on natural language requests. Only generate SELECT statements. "
        "Use the database schema to form correct SQL.\n\n"
    )
    schema_context = f"Database schema:\n{schema_description}\n\n"
    few_shot_examples = (
        "Example 1:\n"
        "Request: List all customers from Germany\n"
        "SQL: SELECT * FROM customers WHERE country = 'Germany';\n\n"
        "Example 2:\n"
        "Request: Show product names and prices cheaper than 50\n"
        "SQL: SELECT product_name, price FROM products WHERE price < 50;\n\n"
    )
    prompt = system_instructions + schema_context + few_shot_examples + f"Request: {user_request}\nSQL:"
    return prompt

# -------------------------------
# Main Interactive Loop

async def main(db_path: str):
    schema_desc = extract_schema_sqlite(db_path)
    if not schema_desc:
        logger.error("Failed to retrieve schema description. Exiting.")
        return

    nl_sql_model = NLToSQLModel()
    sql_executor = SQLExecutor(db_path)

    print("\n=== Natural Language to SQL Bot ===")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter your request: ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        prompt = build_prompt(schema_desc, user_input)
        logger.debug(f"Prompt for model:\n{prompt}")

        sql_query = nl_sql_model.generate_sql(prompt)
        print(f"\nGenerated SQL:\n{sql_query}")

        print("Executing SQL on the database (auto-confirmed)...")
        result = sql_executor.execute_sql(sql_query)
        if isinstance(result, tuple):
            rows, headers = result
            pretty_print_result(rows, headers)
        else:
            print(result)

# -------------------------------
# Entry Point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natural Language to SQL Bot")
    parser.add_argument("--db", default="mydata.db", help="Path to SQLite database file")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.db))
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting...")