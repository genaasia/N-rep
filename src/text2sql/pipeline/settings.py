from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class GeneratorConfig:
    name: str
    model: str
    top_k: int
    config: Dict[str, Any] | None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "model": self.model, "config": self.config, "top_k":self.top_k}


@dataclass
class PipeConfig:
    formatter: str
    schema: str
    postfunc: str
    generator: GeneratorConfig
    candidate_count: int
    pipe_name: str
    rewrite: bool = False
    repair: bool = False
    add_date: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "rewrite": self.rewrite,
            "repair": self.repair,
            "add_date": self.add_date,
            "schema": self.schema,
            "postfunc": self.postfunc,
            "generator": self.generator.to_dict(),
            "candidate_count": self.candidate_count,
            "pipe_name": self.pipe_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipeConfig":
        generator_data = data["generator"]
        generator = GeneratorConfig(
            name=generator_data["name"],
            model=generator_data["model"],
            top_k=generator_data["top_k"],
            config=generator_data.get("config"),
        )
        return cls(
            formatter=data["formatter"],
            rewrite=data.get("rewrite", False),
            repair=data.get("repair", False),
            add_date=data.get("add_date", True),
            schema=data["schema"],
            postfunc=data["postfunc"],
            generator=generator,
            candidate_count=data["candidate_count"],
            pipe_name=data["pipe_name"],
        )


@dataclass
class Settings:
    log_folder: Path
    outputs_folder: Path
    results_folder: Path
    plots_folder: Path
    inference_folder: Path
    train_file_path: Path
    train_embedding_file_path: Path
    test_file_path: Path
    collection_name: str
    database_type: str
    question_key: str
    target_sql_key: str
    db_name_key: str
    benchmark: bool
    batch_size: int
    max_workers: int
    pipe_configurations: List[PipeConfig]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Load settings from a YAML file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Convert pipe configurations to PipeConfig objects
        pipe_configs = [
            PipeConfig.from_dict(pipe_config)
            for pipe_config in config["pipe_configurations"]
        ]

        return cls(
            log_folder=Path(config["log_folder"]),
            outputs_folder=Path(config["outputs_folder"]),
            results_folder=Path(config["results_folder"]),
            inference_folder=Path(config["inference_folder"]),
            plots_folder=Path(config["plots_folder"]),
            train_file_path=Path(config["train_file_path"]),
            train_embedding_file_path=Path(config["train_embedding_file_path"]),
            test_file_path=Path(config["test_file_path"]),
            collection_name=config["collection_name"],
            database_type=config["database_type"],
            question_key=config["question_key"],
            target_sql_key=config["target_sql_key"],
            db_name_key=config["db_name_key"],
            benchmark=config["benchmark"],
            batch_size=config["batch_size"],
            max_workers=config["max_workers"],
            pipe_configurations=pipe_configs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to a dictionary."""
        return {
            "log_folder": str(self.log_folder),
            "outputs_folder": str(self.outputs_folder),
            "results_folder": str(self.results_folder),
            "train_file_path": str(self.train_file_path),
            "train_embedding_file_path": str(self.train_embedding_file_path),
            "test_file_path": str(self.test_file_path),
            "collection_name": str(self.collection_name),
            "database_type": str(self.database_type),
            "question_key": str(self.question_key),
            "target_sql_key": str(self.target_sql_key),
            "db_name_key": str(self.db_name_key),
            "benchmark": str(self.benchmark),
            "batch_size": self.batch_size,
            "pipe_configurations": [
                {
                    "formatter": pc.formatter,
                    "schema": pc.schema,
                    "generator": {
                        "name": pc.generator.name,
                        "config": pc.generator.config,
                    },
                    "candidate_count": pc.candidate_count,
                    "pipe_name": pc.pipe_name,
                }
                for pc in self.pipe_configurations
            ],
        }
