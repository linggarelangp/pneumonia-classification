from src.config.load_config import load_config
from pathlib import Path
import os

cfg = load_config()
print(cfg)

path = Path(__file__).resolve().parents[0]
full_path = os.path.join(path, "data", "processed", "chest_xray_processed").replace(os.sep, "/")
files = os.listdir(full_path)

print(files)
print(full_path)
print(os.path.exists(full_path))
print(len(list(Path(full_path).rglob('*'))))