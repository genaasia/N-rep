import yaml


class Config:
    def __init__(self, config_file="config.yaml"):
        """
        Initialize Config object with settings from YAML file

        Args:
            config_file (str): Path to the YAML configuration file
        """
        try:
            with open(config_file, "r") as file:
                config_dict = yaml.safe_load(file)

            # Validate required fields
            required_fields = ["db_name", "data_folder", "schema_file", "subset"]
            for field in required_fields:
                if field not in config_dict:
                    raise KeyError(f"Missing required field: {field}")

            # Set each config item as an attribute
            for key, value in config_dict.items():
                setattr(self, key, value)

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
