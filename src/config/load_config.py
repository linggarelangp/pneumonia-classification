from pathlib import Path
import yaml

def load_config():
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    