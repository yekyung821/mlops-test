# data-prepare/main.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from crawler import fetch_games
from preprocessing import normalize_games, generate_synthetic_users

# Airflow DAG에서 넘겨주는 SHARED_DIR(기본: /opt/shared)
SHARED = Path(os.getenv("SHARED_DIR", "/opt/shared"))
SHARED.mkdir(parents=True, exist_ok=True)

def main():
    # 1) 크롤링
    results = fetch_games()
    # 2) 전처리
    df = normalize_games(results)
    # 3) 가상 로그 생성(필요 시 하이퍼파라미터 조정)
    user_df = generate_synthetic_users(
        df, num_users=100, max_games=5, alpha=0.5, noise_scale=0.1, random_state=42
    )

    # 확인용 출력(옵션)
    try:
        print(df.head())
        print(user_df.head())
    except Exception:
        pass

    # 4) 공용 산출물 경로로 저장
    (SHARED / "popular_games.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SHARED / "popular_games.csv", index=False, encoding="utf-8-sig")
    user_df.to_csv(SHARED / "games_log.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
