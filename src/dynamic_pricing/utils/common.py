from pathlib import Path

import yaml


def read_yaml(path_to_yaml: str | Path) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.
    """
    yaml_path = Path(path_to_yaml)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML content to be a mapping: {yaml_path}")

    return data
