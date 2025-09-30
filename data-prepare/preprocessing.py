import pandas as pd
import numpy as np

# -----------------------------
# 1. RAWG API 데이터 전처리
# -----------------------------
def normalize_games(results):
    rows = []
    for g in results:
        # 장르 첫 항목
        genres = g.get("genres", []) or []
        first_genre = genres[0]["name"] if genres else None

        # 태그 첫 항목의 games_count
        tags = g.get("tags", []) or []
        first_games_count = tags[0].get("games_count") if tags else None

        # 인기도/보유 데이터
        added = g.get("added") or 0
        owned = (g.get("added_by_status") or {}).get("owned") or 0
        owned_ratio = round(owned / added, 2) if added else None

        rows.append({
            "game_id": g.get("id"),
            "game_name": g.get("name"),
            "playtime": g.get("playtime"),
            "rating": g.get("rating"),
            "genre": first_genre,
            "owned_ratio": owned_ratio,
        })

    # 데이터프레임으로 반환
    return pd.DataFrame(rows)


# -----------------------------
# 2. 가상 유저 데이터 생성
# -----------------------------
def generate_synthetic_users(df, num_users=100, max_games=5, alpha=0.5, noise_scale=0.1, random_state=42):
    """
    유저별 선택 다양성을 극대화한 가상 유저 데이터 생성
    - alpha: 가중치 기반 확률 비율 (1-alpha는 uniform)
    - noise_scale: 노이즈 크기 (0~1, 값이 클수록 낮은 확률 게임도 선택될 가능성 증가)
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df["weight"] = df["rating"] * df["owned_ratio"]
    
    base_prob = df["weight"] / df["weight"].sum()
    uniform_prob = np.ones(len(df)) / len(df)
    
    user_data = []
    
    for user_id in range(1, num_users + 1):
        # 유저별 확률 생성
        noise = rng.normal(0, noise_scale, len(df))  # 큰 노이즈 추가
        probs = alpha * base_prob + (1 - alpha) * uniform_prob + noise
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        
        n_games = rng.integers(1, max_games + 1)
        chosen_games = rng.choice(
            df.index,
            size=n_games,
            replace=False,
            p=probs
        )
        
        for idx in chosen_games:
            row = df.loc[idx]
            user_data.append({
                "user_id": user_id,
                "game_id": row["game_id"],
                "game_name": row["game_name"]
            })
    
    return pd.DataFrame(user_data)
