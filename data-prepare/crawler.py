import os
import requests

def fetch_games(
    tags: str = "free-to-play",
    genres: str = "action",
    page_size: int = 40,
    ordering: str = "-rating",
    base_url: str | None = None,
    api_key: str | None = None,
):

    # 환경변수
    base_url = base_url or os.getenv("RAWG_BASE_URL", "https://api.rawg.io/api/games")
    api_key  = api_key  or os.getenv("RAWG_API_KEY")

    if not api_key:
        raise ValueError("RAWG_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")

    # 파라미터
    params = {
        "key": api_key,
        "tags": tags,
        "genres": genres,
        "page_size": page_size,
        "ordering": ordering,
    }

    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])
