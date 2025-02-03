import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration from YAML file")
    parser.add_argument(
        "--config",
        default="config.yaml",
        type=str,
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()
