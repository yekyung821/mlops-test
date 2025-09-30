from pathlib import Path
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import TemplateNotFound
from fastapi.templating import Jinja2Templates

# ---- 경로 유틸(환경변수 → /opt → 로컬 fallback) ----
import os
def _shared_dir() -> Path:
    v = os.getenv("SHARED_DIR")
    if v:
        return Path(v)
    d = Path("/opt/shared")
    return d if d.exists() else Path(__file__).resolve().parents[3] / "shared"

def _reco_path() -> Path:
    v = os.getenv("RECO_PATH")
    return Path(v) if v else (_shared_dir() / "recommendations.json")

SHARED_DIR = _shared_dir()
RECO_PATH = _reco_path()

APP_DIR = Path(__file__).parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

def _load_recommendations(user_id: int, top_k: int) -> List[str]:
    """recommendations.json에서 해당 user의 추천 리스트를 읽고, 없으면 빈 리스트"""
    try:
        if RECO_PATH.exists():
            data = json.loads(RECO_PATH.read_text(encoding="utf-8"))
            names = data.get(str(user_id)) or data.get(int(user_id)) or []
            names = [str(x) for x in names][:top_k]
            return names
    except Exception:
        pass
    return []

def _fallback_popular(top_k: int) -> List[str]:
    """popular_games.csv 상위 N개로 폴백"""
    csv_path = SHARED_DIR / "popular_games.csv"
    try:
        import pandas as pd
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for key in ("game_name", "title", "name"):
                if key in df.columns:
                    return df[key].astype(str).head(top_k).tolist()
            return df.iloc[:, 0].astype(str).head(top_k).tolist()
    except Exception:
        pass
    return []

def _decorate(names: List[str]) -> List[Dict[str, Any]]:
    """템플릿에 넘기기 좋은 구조로 가공 (이미지가 있으면 붙여줌)"""
    items = []
    for n in names:
        # 정직하게 이름 기준 jpg 찾아보기(있으면 써주고, 없으면 None)
        img = None
        candidate = (APP_DIR / "static" / f"{n}.jpg")
        if candidate.exists():
            img = f"/static/{n}.jpg"
        items.append({"name": n, "img": img})
    return items

@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    user_id: int = Query(5, description="User ID"),
    top_k: int = Query(5, description="Top-K"),
):
    names = _load_recommendations(user_id, top_k)
    if not names:
        names = _fallback_popular(top_k)

    games = _decorate(names)
    ctx = {"request": request, "user_id": user_id, "top_k": top_k, "games": games}

    # 기존 템플릿이 있으면 사용, 없으면 매우 간단한 HTML로 응답
    try:
        return templates.TemplateResponse("index.html", ctx)
    except TemplateNotFound:
        html_items = "".join(
            f"<li>{g['name']}" + (f"<br><img src='{g['img']}' width='200'>" if g['img'] else "") + "</li>"
            for g in games
        )
        html = f"""
        <html><head><title>게임 추천</title></head><body>
        <h1>게임 추천 시스템</h1>
        <form method="get" action="/">
          User ID: <input type="number" name="user_id" value="{user_id}">
          Top K: <input type="number" name="top_k" value="{top_k}">
          <button type="submit">추천 받기</button>
        </form>
        <h2>User {user_id} 추천 결과:</h2>
        <ul>{html_items or "<li>(추천 없음)</li>"}</ul>
        </body></html>
        """
        return HTMLResponse(content=html)
