from pathlib import Path
import os
import sys

# Define directories
current_path = os.path.abspath(".")
root_name = "data-lifecycle"
ROOT = Path(current_path[: current_path.find(root_name) + len(root_name)])

DATASET_DIR = ROOT / "datasets"
FIGURES_DIR = ROOT / "data_analysis" / "figures"

# Create directories
DATASET_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
