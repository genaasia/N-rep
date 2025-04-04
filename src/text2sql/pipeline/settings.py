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
    use_evidence: bool = False
    schema_key: str = ""
    fewshot_schema_key: str = "gold_filtered_schema"  # Default value for backward compatibility

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formatter": self.formatter,
            "rewrite": self.rewrite,
            "repair": self.repair,
            "add_date": self.add_date,
            "schema": self.schema,
            "postfunc": self.postfunc,
            "generator": self.generator.to_dict(),
            "candidate_count": self.candidate_count,
            "pipe_name": self.pipe_name,
            "use_evidence": self.use_evidence,
            "schema_key": self.schema_key,
            "fewshot_schema_key": self.fewshot_schema_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipeConfig":
        generator_data = data["generator"]
        generator = GeneratorConfig(
            name=generator_data["name"],
            model=generator_data["model"],
            top_k=generator_data.get("top_k", 0),
            config=generator_data.get("config"),
        )
        return cls(
            formatter=data["formatter"],
            rewrite=data.get("rewrite", False),
            repair=data.get("repair", False),
            add_date=data.get("add_date", True),
            schema_key=data.get("schema_key", ""),
            use_evidence=data.get("use_evidence", False),
            schema=data["schema"],
            postfunc=data["postfunc"],
            generator=generator,
            candidate_count=data["candidate_count"],
            pipe_name=data["pipe_name"],
            fewshot_schema_key=data.get("fewshot_schema_key", "gold_filtered_schema"),
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
    metrics: List[str] = None  # Default to None to maintain backward compatibility

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Load settings from a YAML file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract values from nested structure
        inputs = config.get("inputs", {})
        outputs = config.get("outputs", {})
        data = config.get("data", {})
        processing = config.get("processing", {})
        evaluation = config.get("evaluation", {})
        
        # Convert pipe configurations to PipeConfig objects
        pipe_configs = [
            PipeConfig.from_dict(pipe_config)
            for pipe_config in config.get("pipe_configurations", [])
        ]

        return cls(
            # Outputs section
            log_folder=Path(outputs.get("log_folder", config.get("log_folder", "./logs"))),
            outputs_folder=Path(outputs.get("outputs_folder", config.get("outputs_folder", "./outputs"))),
            results_folder=Path(outputs.get("results_folder", config.get("results_folder", "./results"))),
            plots_folder=Path(outputs.get("plots_folder", config.get("plots_folder", "./plots"))),
            inference_folder=Path(outputs.get("inference_folder", config.get("inference_folder", "./inference"))),
            
            # Data section
            database_type=data.get("database_type", config.get("database_type", "sqlite")),
            collection_name=data.get("collection_name", config.get("collection_name", "default_collection")),
            question_key=data.get("question_key", config.get("question_key", "question")),
            target_sql_key=data.get("target_sql_key", config.get("target_sql_key", "SQL")),
            db_name_key=data.get("db_name_key", config.get("db_name_key", None)),
            benchmark=data.get("benchmark", config.get("benchmark", False)),

            # Inputs section
            train_file_path=Path(inputs.get("train_file_path", config.get("train_file_path", ""))),
            train_embedding_file_path=Path(inputs.get("train_embedding_file_path", config.get("train_embedding_file_path", ""))),
            test_file_path=Path(inputs.get("test_file_path", config.get("test_file_path", ""))),
            
            # Processing section
            batch_size=processing.get("batch_size", config.get("batch_size", 1)),
            max_workers=processing.get("max_workers", config.get("max_workers", 1)),
            
            # Pipeline configurations
            pipe_configurations=pipe_configs,
            
            # Evaluation section
            metrics=evaluation.get("metrics", ["sql_match", "execution_match", "intent", "soft_f1"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to a nested dictionary."""
        return {
            "inputs": {
                "train_file_path": str(self.train_file_path),
                "train_embedding_file_path": str(self.train_embedding_file_path),
                "test_file_path": str(self.test_file_path),
            },
            "outputs": {
                "log_folder": str(self.log_folder),
                "outputs_folder": str(self.outputs_folder),
                "results_folder": str(self.results_folder),
                "plots_folder": str(self.plots_folder),
                "inference_folder": str(self.inference_folder),
            },
            "data": {
                "type": self.database_type,
                "collection_name": self.collection_name,
                "question_key": self.question_key,
                "target_sql_key": self.target_sql_key,
                "benchmark": self.benchmark,
                "db_name_key": self.db_name_key,
            },
            "processing": {
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
            },
            "evaluation": {
                "metrics": self.metrics,
            },
            "pipe_configurations": [pc.to_dict() for pc in self.pipe_configurations],
        }
