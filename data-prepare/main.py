# mlops/src/main.py
import os
import sys
import subprocess
from pathlib import Path

# ------- 환경변수 (Airflow에서 넣어줌) -------
# 공유 디렉토리(볼륨)
SHARED_DIR = Path(os.getenv("SHARED_DIR", "/opt/shared"))
# 학습 산출물 디렉토리
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(SHARED_DIR / "model")))
# 데이터 경로(우선 /opt/shared/games_log.csv, 없으면 레포 내 기본 파일로 폴백)
GAMES_LOG_PATH = os.getenv("GAMES_LOG_PATH", str(SHARED_DIR / "games_log.csv"))
# Weights & Biases
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_MODE = os.getenv("WANDB_MODE", "online")  # online/offline

def _prepare_paths() -> str:
    """필요한 디렉토리 만들고, 데이터 경로를 최종 확정해서 반환."""
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path(GAMES_LOG_PATH)
    if not data_path.exists():
        fallback = Path("/opt/mlops/dataset/games_log.csv")
        if fallback.exists():
            print(f"[WARN] {data_path} 가 없어 기본 데이터로 대체: {fallback}", file=sys.stderr)
            return str(fallback)
        else:
            print(f"[WARN] 데이터 파일을 찾을 수 없습니다: {data_path}", file=sys.stderr)
    return str(data_path)

def run_train():
    data_path = _prepare_paths()

    # 서브프로세스에 넘길 환경
    env = os.environ.copy()
    env["MODEL_DIR"] = str(MODEL_DIR)
    env["GAMES_LOG_PATH"] = data_path
    if WANDB_API_KEY:
        env["WANDB_API_KEY"] = WANDB_API_KEY
    env.setdefault("WANDB_MODE", WANDB_MODE)

    # ✅ 방어용: src.* 임포트가 어디서든 되도록 PYTHONPATH 보정
    env["PYTHONPATH"] = f"/opt/mlops:{env.get('PYTHONPATH','')}"

    # train 스크립트를 파일 경로로 실행 (패키지 임포트/워크디렉터리 이슈 회피)
    script_path = Path(__file__).resolve().parent / "train" / "train.py"
    cmd = ["python", str(script_path)]
    print(f"[INFO] Running: {' '.join(cmd)}")
    print(f"[INFO] GAMES_LOG_PATH={data_path}")
    print(f"[INFO] MODEL_DIR={MODEL_DIR}")

    try:
        res = subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed with return code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    print("[INFO] Training finished successfully.")

if __name__ == "__main__":
    run_train()
