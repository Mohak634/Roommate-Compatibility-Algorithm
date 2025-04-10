from pathlib import Path

def get_data_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "data" / filename

def get_model_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "models" / filename


