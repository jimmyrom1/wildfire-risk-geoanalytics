"""
Descarga el dataset público "Forest Fires" (UCI Machine Learning Repository,
Cortez & Morais, 2007) y lo guarda en data/raw/forestfires.csv.

Uso:
    python -m etl.download_data
"""
from pathlib import Path

import requests

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "forestfires.csv"


def download(url: str = UCI_URL, output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


if __name__ == "__main__":
    path = download()
    print(f"Dataset descargado en: {path}")
