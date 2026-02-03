"""Project configuration (paths and random seed).

Keeping config in one place makes the project easier to read and change.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    # Base folder = project root (two levels up from this file)
    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw" / "insurance.csv"
    data_processed: Path = root / "data" / "processed" / "insurance_clean.csv"
    models_dir: Path = root / "models"
    reports_figures: Path = root / "reports" / "figures"
    reports_metrics: Path = root / "reports" / "metrics"


RANDOM_SEED = 42
