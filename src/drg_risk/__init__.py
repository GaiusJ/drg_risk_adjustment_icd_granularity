from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Daten" # Speicherort der Daten
SCORES_DIR = DATA_DIR / "scores"  # Speicherort der Metriken
REPORT_DIR = DATA_DIR / "reports" # Speicherort der Berichte