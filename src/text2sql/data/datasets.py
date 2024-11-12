import glob
import os

from abc import ABC, abstractmethod

import sqlalchemy

from sqlalchemy import create_engine, inspect, text

from text2sql.data.sqlite_functions import  get_sqlite_database_file, query_sqlite_database, get_sqlite_schema
from text2sql.data.schema_to_text import schema_to_basic_format, schema_to_sql_create, schema_to_datagrip_format
from text2sql.data.mysql_functions import get_mysql_schema
from text2sql.data.postgres_functions import get_postgresql_schema

def list_supported_databases(dataset_base_path: str) -> list[str]:
    """find all sqlite databases in the dataset directory and return their names"""
    # handle nested or flat structure
    flat = [os.path.basename(p) for p in glob.glob(os.path.join(dataset_base_path, "*.sqlite"))]
    nested = [os.path.basename(p) for p in glob.glob(os.path.join(dataset_base_path, "**/*.sqlite"))]
    found_files = sorted(list(set(flat + nested)))
    database_names = [x.rsplit(".", 1)[0] for x in found_files]
    return database_names


class BaseDataset(ABC):
    @abstractmethod
    def get_databases(self) -> list[str]:
        pass
    
    @abstractmethod
    def get_schema_description_modes(self) -> list[str]:
        pass
    
    @abstractmethod
    def get_database_schema(self, database_name: str) -> dict:
        pass
    
    @abstractmethod
    def describe_database_schema(self, database_name: str, mode: str="basic") -> str:
        pass
    
    @abstractmethod
    def query_database(self, database_name: str, query: str) -> list[dict]:
        pass

    def validate_query(self, database_name: str, query: str) -> dict:
        """validate the query against the database schema"""
        try:
            result: list[dict] = self.query_database(database_name, query)
            success: bool = True
            message: str = "ok"
        except Exception as e:
            result: list[dict] = []
            success: bool = False
            message: str = f"error - {type(e).__name__}: {str(e)}"
        return {"validated": success, "message": message, "execution_result": result}


class MysqlDataset(BaseDataset):
    def __init__(self, host: str, port: int, user: str, password: str):
        """initialize a mysql dataset manager"""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.supported_modes = ["basic", "basic_types", "basic_relations", "basic_types_relations", "sql", "datagrip"]
        self.engine = self._get_engine()
        self.databases = self.get_databases()

    def _get_connection_string(self) -> list[str]:
        """return connection string to database (PyMySQL)"""
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}"
    
    def _get_engine(self) -> sqlalchemy.engine.Engine:
        """return the path to the sqlite database file"""
        connection_string = self._get_connection_string()
        return create_engine(connection_string)
    
    def get_databases(self) -> list[str]:
        """get a list of databases"""
        inspector = inspect(self.engine)
        return inspector.get_schema_names()

    def get_schema_description_modes(self) -> list[str]:    
        """return a list of the supported schema modes"""
        return self.supported_modes
    
    def get_database_schema(self, database_name: str) -> dict:
        """return a dict of the database schema"""
        if database_name not in self.databases:
            raise ValueError(f"Database '{database_name}' not in databases {self.databases}")
        return get_mysql_schema(self.engine, database_name)
    
    def describe_database_schema(self, database_name: str, mode: str="basic") -> str:
        """return a string representation of the database schema"""
        if database_name not in self.databases:
            raise ValueError(f"Database '{database_name}' not in databases {self.databases}")
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
        if database_name not in self.databases:
            raise ValueError(f"Database '{database_name}' not in databases {self.databases}")
        with self.engine.connect() as connection:
            connection.execute(text(f"USE {database_name};"))
            result = connection.execute(text(query))
        return [dict(r._mapping) for r in result]
    

class PostgresDataset(BaseDataset):
    def __init__(self, host: str, port: int, user: str, password: str):
        """initialize a postgres dataset manager"""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.supported_modes = ["basic", "basic_types", "basic_relations", "basic_types_relations", "sql", "datagrip"]
        self.engines = dict()  # db name, engine
        self.databases = sorted(list(self.engines.keys()))

    def _get_connection_string(self, database_name: str) -> list[str]:
        """return connection string to database (psycopg2)"""
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{database_name}"
    
    def _get_engine(self, database_name: str) -> sqlalchemy.engine.Engine:
        """return the path to the sqlite database file"""
        if database_name not in self.databases:
            connection_string = self._get_connection_string(database_name)
            self.engines[database_name] = create_engine(connection_string)
            self.get_databases()
        return self.engines[database_name]
    
    def get_databases(self) -> list[str]:
        """get a list of databases"""
        self.databases = sorted(list(self.engines.keys()))
        return self.databases

    def get_schema_description_modes(self) -> list[str]:    
        """return a list of the supported schema modes"""
        return self.supported_modes
    
    def get_database_schema(self, database_name: str) -> dict:
        """return a dict of the database schema"""
        engine = self._get_engine(database_name)
        return get_postgresql_schema(engine, database_name)
    
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
        engine = self._get_engine(database_name)
        with engine.connect() as connection:
            result = connection.execute(text(query))
        return [dict(r._mapping) for r in result]
    

class SqliteDataset(BaseDataset):
    def __init__(self, base_data_path: str):
        """initialize an sql dataset manager
        
        list, describe and query sqlite databases from sqlite based datasets.  
        the base path should be the main directory of the databases,  
        e.g. for BIRD, "<my_path_to>/bird/train/train_databases"

        Args:
            base_data_path (str): the base path of the dataset containing the databases
        """
        self.base_data_path = base_data_path
        self.databases = list_supported_databases(base_data_path)
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
            raise ValueError(f"Database '{database_name}' not found in '{self.base_data_path}'")
        return get_sqlite_database_file(self.base_data_path, database_name)
    
    def get_database_schema(self, database_name: str) -> dict:
        """return a dict of the database schema"""
        return get_sqlite_schema(self.base_data_path, database_name)
    
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
        return query_sqlite_database(self.base_data_path, database_name, query)


class SqliteDataset:
    def __init__(self, base_data_path: str):
        """initialize an sql dataset manager
        
        list, describe and query sqlite databases from sqlite based datasets.  
        the base path should be the main directory of the databases,  
        e.g. for BIRD, "<my_path_to>/bird/train/train_databases"

        Args:
            base_data_path (str): the base path of the dataset containing the databases
        """
        self.base_data_path = base_data_path
        self.databases = list_supported_databases(base_data_path)
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
            raise ValueError(f"Database '{database_name}' not found in '{self.base_data_path}'")
        return get_sqlite_database_file(self.base_data_path, database_name)
    
    def get_database_schema(self, database_name: str) -> dict:
        """return a dict of the database schema"""
        return get_sqlite_schema(self.base_data_path, database_name)
    
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
        return query_sqlite_database(self.base_data_path, database_name, query)