from pathlib import Path
import os

# Define directories
ROOT = Path(os.path.abspath(".."))
DATASET_DIR = ROOT / "datasets"

FIGURES_DIR = ROOT / "data_analysis" / "figures"

# Create directories
DATASET_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
