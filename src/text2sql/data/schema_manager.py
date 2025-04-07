from typing import Dict, List, Optional
import json
from pathlib import Path
from text2sql.data.datasets import BaseDataset
from text2sql.data.schema_filtering import (
    parse_mac_schema,
    parse_m_schema,
    parse_sql_create,
    parse_basic_format,
    parse_datagrip_format,
)


class SchemaManager:
    """Manages schema descriptions for all databases in a dataset.

    This class maintains in-memory copies of schema descriptions for all databases
    in a dataset, in various formats. It also provides methods to get filtered
    versions of these schemas.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        supported_modes: Optional[List[str]] = None,
        table_descriptions_path: Optional[str] = None,
    ):
        """Initialize the SchemaManager.

        Args:
            dataset: The dataset containing the databases
            supported_modes: List of schema description modes to support.
                           If None, uses all modes supported by the dataset.
            table_descriptions_path: Path to JSON file containing table descriptions.
                                   If provided, enables full mac-schema mode with descriptions.
                                   If None, uses mac-schema-basic mode without descriptions.
        """
        self.dataset = dataset
        self.supported_modes = supported_modes or dataset.supported_modes

        # Load table descriptions if provided
        self.table_descriptions = None
        if table_descriptions_path is not None:
            with open(table_descriptions_path, "r") as f:
                self.table_descriptions = json.load(f)

        # Validate that all requested modes are supported by the dataset
        for mode in self.supported_modes:
            if mode not in dataset.supported_modes:
                raise ValueError(f"Mode '{mode}' is not supported by the dataset")

        # Initialize schema storage
        self.schema_maps: Dict[str, Dict] = {}
        self.schemas: Dict[str, Dict[str, str]] = {}

        # Load schemas for all databases and modes
        self._load_schemas()

    def _get_table_descriptions(self, db_name: str) -> Optional[Dict]:
        """Get table descriptions for a specific database.

        Args:
            db_name: Name of the database

        Returns:
            Table descriptions dict if found, None otherwise
        """
        if self.table_descriptions is None:
            return None

        for desc in self.table_descriptions:
            if desc.get("db_id") == db_name:
                return desc
        return None

    def _get_effective_mode(self, mode: str, db_name: str) -> str:
        """Get the effective mode to use, handling mac-schema variants.

        Args:
            mode: Requested mode
            db_name: Name of the database

        Returns:
            Effective mode to use
        """
        if mode == "mac_schema":
            if self.table_descriptions is not None and self._get_table_descriptions(db_name) is not None:
                return "mac_schema"
            return "mac_schema_basic"
        return mode

    def _load_schemas(self):
        """Load schema descriptions for all databases and modes."""
        for db_name in self.dataset.get_databases():
            self.schema_maps[db_name] = self.dataset.get_database_schema(db_name)
            self.schemas[db_name] = {}
            for mode in self.supported_modes:
                effective_mode = self._get_effective_mode(mode, db_name)

                # Handle mac-schema mode specially
                if effective_mode == "mac_schema":
                    table_descriptions = self._get_table_descriptions(db_name)
                    schema = self.dataset.describe_database_schema(
                        db_name, effective_mode, table_descriptions=table_descriptions
                    )
                else:
                    schema = self.dataset.describe_database_schema(db_name, effective_mode)

                # Store under the original mode name for consistent access
                self.schemas[db_name][mode] = schema

    def get_filtered_schema(self, database_name: str, filter_dict: Dict[str, List[str]], mode: str) -> str:
        """Get a filtered schema description for a database.

        Args:
            database_name: Name of the database
            filter_dict: Dictionary mapping table names to lists of columns to keep
            mode: Schema description mode to use

        Returns:
            Filtered schema description string

        Raises:
            ValueError: If database_name or mode is not found
        """
        if database_name not in self.schemas:
            raise ValueError(f"Database '{database_name}' not found")
        if mode not in self.schemas[database_name]:
            raise ValueError(f"Mode '{mode}' not found for database '{database_name}'")

        full_schema = self.schemas[database_name][mode]

        # Apply appropriate filtering function based on mode
        if mode == "mac_schema":
            return parse_mac_schema(full_schema, filter_dict)
        elif mode == "m_schema":
            return parse_m_schema(full_schema, filter_dict)
        elif mode == "sql":
            return parse_sql_create(full_schema, filter_dict)
        elif mode in ["basic", "basic_types", "basic_relations", "basic_types_relations"]:
            include_types = "types" in mode
            include_relations = "relations" in mode
            return parse_basic_format(full_schema, filter_dict, include_types, include_relations)
        elif mode == "datagrip":
            return parse_datagrip_format(full_schema, filter_dict)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def get_schema_mapping(self, database_name: str) -> Dict:
        """Get the schema mapping for a database.

        Args:
            database_name: Name of the database

        Returns:
            Schema mapping dictionary

        Raises:
            ValueError: If database_name is not found
        """
        if database_name not in self.schema_maps:
            raise ValueError(f"Database '{database_name}' not found")
        return self.schema_maps[database_name]

    def get_full_schema(self, database_name: str, mode: str) -> str:
        """Get the full schema description for a database.

        Args:
            database_name: Name of the database
            mode: Schema description mode to use

        Returns:
            Full schema description string

        Raises:
            ValueError: If database_name or mode is not found
        """
        if database_name not in self.schemas:
            raise ValueError(f"Database '{database_name}' not found")
        if mode not in self.schemas[database_name]:
            raise ValueError(f"Mode '{mode}' not found for database '{database_name}'")

        return self.schemas[database_name][mode]
