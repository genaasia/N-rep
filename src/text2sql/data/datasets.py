import glob
import os

from text2sql.data.sqlite_functions import  get_sqlite_database_file, query_sqlite_database, get_sqlite_schema
from text2sql.data.schema_to_text import schema_to_basic_format, schema_to_sql_create, schema_to_datagrip_format


def list_supported_databases(dataset_base_path: str, dataset: str) -> list[str]:
    """find all sqlite databases in the dataset directory and return their names"""
    # handle nested or flat structure
    flat = [os.path.basename(p) for p in glob.glob(os.path.join(dataset_base_path, dataset, "*.sqlite"))]
    nested = [os.path.basename(p) for p in glob.glob(os.path.join(dataset_base_path, dataset, "**/*.sqlite"))]
    found_files = sorted(list(set(flat + nested)))
    database_names = [x.rsplit(".", 1)[0] for x in found_files]
    return database_names


class SqliteDataset:
    def __init__(self, base_data_path: str, dataset_name: str):
        self.base_data_path = base_data_path
        self.dataset_name = dataset_name
        self.databases = list_supported_databases(base_data_path, dataset_name)
        self.supported_modes = ["basic", "basic_types", "basic_relations", "basic_types_relations", "sql", "datagrip"]

    def get_databases(self) -> list[str]:
        """return a list of the names of the sqlite databases in the dataset"""
        return self.databases
    
    def get_schema_description_modes(self) -> list[str]:    
        """return a list of the supported schema modes"""
        return self.supported_modes

    def get_database_path(self, database_name: str) -> str:
        """return the path to the sqlite database file"""
        if database_name not in self.databases:
            raise ValueError(f"Database '{database_name}' not found in dataset '{self.dataset_name}'")
        return get_sqlite_database_file(self.base_data_path, self.dataset_name, database_name)
    
    def get_database_schema(self, database_name: str) -> dict:
        """return a dict of the database schema"""
        return get_sqlite_schema(self.base_data_path, self.dataset_name, database_name)
    
    def describe_database_schema(self, database_name: str, mode: str="basic") -> str:
        """return a string representation of the database schema"""
        
        if mode not in self.supported_modes:
            raise ValueError(f"Unknown schema mode '{mode}', supported modes are: {self.supported_modes}")
        schema = self.get_database_schema(database_name)
        if mode == "basic":
            return schema_to_basic_format(database_name, schema, include_types=False, include_relations=False)
        if mode == "basic_types":
            return schema_to_basic_format(database_name, schema, include_types=True, include_relations=False)
        if mode == "basic_relations":
            return schema_to_basic_format(database_name, schema, include_types=False, include_relations=True)
        if mode == "basic_types_relations":
            return schema_to_basic_format(database_name, schema, include_types=True, include_relations=True)
        elif mode == "sql":
            return schema_to_sql_create(database_name, schema)
        elif mode == "datagrip":
            return schema_to_datagrip_format(database_name, schema)
        else:
            raise ValueError(f"Unknown schema mode '{mode}', supported modes are: {self.supported_modes}")
        
    def query_database(self, database_name: str, query: str) -> list[dict]:
        """return the results of the query as a list of dictionaries"""
        return query_sqlite_database(self.base_data_path, self.dataset_name, database_name, query)