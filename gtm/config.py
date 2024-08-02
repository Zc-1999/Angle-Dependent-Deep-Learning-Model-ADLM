from typing import Dict

import yaml


def read_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        task_config = yaml.safe_load(f)
    return task_config
