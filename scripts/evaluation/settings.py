from typing import Dict, Any
import yaml
from dataclasses import dataclass


@dataclass
class EvalConfig:
    file_name: str
    out_folder: str
    plot_folder: str
    results_tag: str
    results_file_name: str
    nlq_col: str
    # user_id_col: str
    ex_label_col: str
    sql_col: str
    template_id_col: str
    pred_ex_label_col: str
    pred_sql_col: str
    pred_template_id_col: str


@dataclass
class Settings:
    url: str
    client: str
    dataset: str
    db_type: str
    db_name: str
    gen_models: list[dict]
    gen_mods: list
    collection: str
    num_of_workers: int
    metrics: list
    plot_labels: list
    eval_data_config: EvalConfig

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Settings":
        eval_data = config["eval_data"]
        eval_data_config = EvalConfig(
            file_name=eval_data["file_name"],
            out_folder=eval_data["out_folder"],
            plot_folder=eval_data["plot_folder"],
            results_tag=eval_data["results_tag"],
            results_file_name=eval_data["results_file_name"],
            nlq_col=eval_data["in"]["nlq_col"],
            # user_id_col=eval_data["in"]["user_id_col"],
            ex_label_col=eval_data["in"]["ex_label_col"],
            sql_col=eval_data["in"]["sql_col"],
            template_id_col=eval_data["in"]["template_id_col"],
            pred_ex_label_col=eval_data["out"]["pred_ex_label_col"],
            pred_sql_col=eval_data["out"]["pred_sql_col"],
            pred_template_id_col=eval_data["out"]["pred_template_id_col"],
        )
        gen_models = config["inference"]["gen_models"]
        for mdl in gen_models:
            if "backend" not in mdl:
                raise KeyError("backend key is missing in gen_models")
            if "model" not in mdl:
                raise KeyError("model key is missing in gen_models")
        return cls(
            url=config["inference"]["url"],
            client=config["inference"]["client"],
            dataset=config["inference"]["dataset"],
            db_type=config["inference"]["db_type"],
            db_name=config["inference"]["db_name"],
            gen_models=gen_models,
            gen_mods=config["inference"]["gen_mods"],
            collection=config["inference"]["collection"],
            num_of_workers=config["inference"]["num_of_workers"],
            metrics=config["metrics"],
            plot_labels=[metric_config["plot_label"] for metric_config in config["metrics"]],
            eval_data_config=eval_data_config,
        )


_settings: Settings | None = None


def load_settings(config_file: str = "config.yaml") -> Settings:
    """Load settings from a specified config file."""
    global _settings

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    _settings = Settings.from_config(config)
    return _settings


def get_settings() -> Settings:
    """Get the current settings instance."""
    if _settings is None:
        return load_settings()
    return _settings


def get_settings_vars(config_file: str = "config.yaml"):
    settings = load_settings(config_file)
    eval_data_config = settings.eval_data_config
    return (
        settings.url,
        settings.gen_models,
        settings.gen_mods,
        settings.num_of_workers,
        eval_data_config.file_name,
        eval_data_config.out_folder,
        eval_data_config.plot_folder,
        eval_data_config.results_tag,
        eval_data_config.results_file_name,
        eval_data_config.nlq_col,
        # eval_data_config.user_id_col,
        eval_data_config.ex_label_col,
        eval_data_config.sql_col,
        eval_data_config.template_id_col,
        eval_data_config.pred_ex_label_col,
        eval_data_config.pred_sql_col,
        eval_data_config.pred_template_id_col,
        settings.metrics,
        settings.plot_labels,
        eval_data_config,
    ), settings
