import pandas as pd
import numpy as np

def create_user_item_matrix(df: pd.DataFrame, user_col='user_id', item_col='game_name'):
    """
    유저-아이템 행렬 생성
    """
    return pd.crosstab(df[user_col], df[item_col])

def train_val_split(user_item_matrix: pd.DataFrame, val_ratio=0.2, seed=42):
    """
    유저별 80/20 Train/Validation Split
    """
    train_matrix = user_item_matrix.copy()
    val_matrix = np.zeros(user_item_matrix.shape, dtype=int)
    rng = np.random.default_rng(seed)
    
    for u in range(user_item_matrix.shape[0]):
        items = np.where(user_item_matrix.iloc[u].values == 1)[0]
        if len(items) > 1:
            val_items = rng.choice(items, size=max(1, int(len(items)*val_ratio)), replace=False)
            train_matrix.iloc[u, val_items] = 0
            val_matrix[u, val_items] = 1

    val_matrix_df = pd.DataFrame(val_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return train_matrix, val_matrix_df
