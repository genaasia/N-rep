import os

import mysql.connector
from dotenv import load_dotenv

from utils import Config, parse_args

load_dotenv()


def create_database(host, user, password, database, schema_file):
    try:
        # Connect and create database if it doesn't exist
        conn = mysql.connector.connect(host=host, user=user, password=password)
        cursor = conn.cursor()

        # Create and use the database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cursor.execute(f"USE {database}")

        # Disable foreign key checks temporarily
        cursor.execute("SET FOREIGN_KEY_CHECKS=0")

        # Read and execute the schema file
        with open(schema_file, "r") as file:
            schema = file.read()
            schema = schema.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
            statements = schema.split(";")

            # Execute each statement separately
            for statement in statements:
                if statement.strip():
                    try:
                        cursor.execute(statement)
                        conn.commit()
                    except mysql.connector.Error as err:
                        print(f"Warning: {err}")

        # Re-enable foreign key checks
        cursor.execute("SET FOREIGN_KEY_CHECKS=1")

        print(f"Database '{database}' created successfully!")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if "conn" in locals() and conn.is_connected():
            cursor.close()
            conn.close()


if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)

    create_database(
        os.environ.get("MYSQL_HOST"),
        os.environ.get("MYSQL_USER"),
        os.environ.get("MYSQL_PASSWORD"),
        config.db_name,
        config.schema_file,
    )
