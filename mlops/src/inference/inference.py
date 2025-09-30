import os
import json
from pathlib import Path
from typing import List

# ==== 환경변수 (Airflow DAG에서 넘겨줌) ====
USER_ID  = int(os.getenv("USER_ID", "5"))
TOP_K    = int(os.getenv("TOP_K", "5"))
RECO_PATH = Path(os.getenv("RECO_PATH", "/opt/shared/recommendations.json"))
SHARED_DIR = Path(os.getenv("SHARED_DIR", "/opt/shared"))

POPULAR_CSV = SHARED_DIR / "popular_games.csv"  # 크롤/전처리 산출물(옵션)

# ---- 유틸 ----
def _coerce_names(items) -> List[str]:
    """여러 형태(list[str], list[dict], pandas.Series/DataFrame 등)를 안전하게 게임 이름 리스트로 변환"""
    try:
        import pandas as pd
        if isinstance(items, pd.Series):
            return items.astype(str).tolist()
        if isinstance(items, pd.DataFrame):
            for key in ("game_name", "title", "name"):
                if key in items.columns:
                    return items[key].astype(str).tolist()
            return items.iloc[:, 0].astype(str).tolist()
    except Exception:
        pass
    if not items:
        return []
    if isinstance(items, list):
        if items and isinstance(items[0], dict):
            names = []
            for r in items:
                n = r.get("title") or r.get("name") or r.get("game_name")
                if n:
                    names.append(str(n))
            return names
        return [str(x) for x in items]
    return [str(items)]

def _merge_and_write(user_id: int, names: List[str], dst: Path):
    payload = {}
    if dst.exists():
        try:
            payload = json.loads(dst.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload[str(user_id)] = list(map(str, names))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 추천 HOOK ----
def _recommend_via_hook(user_id: int, top_k: int) -> List[str]:
    """
    프로젝트의 '정식' 추천 함수를 호출.
    - 예시1) mlops/src/main.py 에 recommend()가 있는 경우 (네 케이스)
    """
    try:
        from mlops.src.main import recommend
        return _coerce_names(recommend(user_id=user_id, top_k=top_k))[:top_k]
    except Exception:
        pass

    # (옵션) 다른 위치의 함수로 폴백하고 싶으면 여기에 추가
    # try:
    #     from mlops.src.model.game_item_cf import recommend_for_user
    #     return _coerce_names(recommend_for_user(user_id, top_k))[:top_k]
    # except Exception:
    #     pass

    return []

def _fallback_from_popular(top_k: int) -> List[str]:
    """HOOK 실패 시 popular_games.csv 상위 N개로 폴백"""
    try:
        import pandas as pd
        if POPULAR_CSV.exists():
            df = pd.read_csv(POPULAR_CSV)
            for key in ("game_name", "title", "name"):
                if key in df.columns:
                    return df[key].astype(str).head(top_k).tolist()
            return df.iloc[:, 0].astype(str).head(top_k).tolist()
    except Exception:
        pass
    return []

if __name__ == "__main__":
    names = _recommend_via_hook(USER_ID, TOP_K)
    if not names:
        names = _fallback_from_popular(TOP_K)
    _merge_and_write(USER_ID, names[:TOP_K], RECO_PATH)
