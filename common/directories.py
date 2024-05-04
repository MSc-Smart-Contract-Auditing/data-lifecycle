from pathlib import Path
import os

# Define directories
ROOT = Path(os.path.abspath(".."))
DATASET_DIR = ROOT / "datasets"

# Create directories
DATASET_DIR.mkdir(exist_ok=True)
