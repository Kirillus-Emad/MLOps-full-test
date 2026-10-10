import yaml
from pathlib import Path


CONFIG_PATH = Path("configs/config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)