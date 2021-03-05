import os
from pathlib import Path
import numpy as np
import pandas as pd

project_dir = Path(__file__).resolve().parents[2]
data_dir = os.path.join(project_dir, "data")
model_dir = os.path.join(project_dir, "models")