from src.utils.utils import project_path
import os
import pandas as pd

def load_games_log(filename: str):
    data_path = os.path.join(project_path(), "dataset", filename)
    df = pd.read_csv(data_path, encoding="utf-8")
    return df
