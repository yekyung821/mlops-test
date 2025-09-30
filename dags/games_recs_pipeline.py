from datetime import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
from airflow.configuration import conf as airflow_conf

# ---------- Slack: 성공 시 콜백 ----------
def slack_success(context):
    url = Variable.get("SLACK_WEBHOOK_URL", default_var=None)
    if not url:
        return
    dag_run = context["dag_run"]
    dag_id = dag_run.dag_id
    run_id = dag_run.run_id
    conf = dag_run.conf or {}

    who = conf.get("who", "unknown")
    user_id = conf.get("user_id", "N/A")
    top_k = conf.get("top_k", "N/A")
    image_repo = Variable.get("IMAGE_REPO", default_var="yeeho0o/mlops-test")
    image_tag = Variable.get("IMAGE_TAG", default_var="latest")

    base_url = airflow_conf.get("webserver", "base_url", fallback="http://localhost:8080")
    run_link = f"{base_url}/dags/{dag_id}/grid?dag_run_id={run_id}"

    msg = (
        f":tada: *{dag_id}* 성공!\n"
        f"• run_id: `{run_id}`\n"
        f"• by: *{who}*\n"
        f"• params: user_id={user_id}, top_k={top_k}\n"
        f"• image: `{image_repo}:{image_tag}`\n"
        f"• details: {run_link}"
    )
    SlackWebhookHook(webhook_url=url).send(text=msg)

# ---------- 이미지/볼륨 ----------
IMAGE_REPO = Variable.get("IMAGE_REPO", default_var="yeeho0o/mlops-test")
IMAGE_TAG = Variable.get("IMAGE_TAG", default_var="latest")
IMAGE = f"{IMAGE_REPO}:{IMAGE_TAG}"
SHARED_VOLUME = "shared-data:/opt/shared"

default_args = {"owner": "mlops", "depends_on_past": False, "retries": 0}

with DAG(
    dag_id="games_recs_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,           # 수동 실행
    catchup=False,
    default_args=default_args,
    tags=["mlops", "recsys"],
    on_success_callback=slack_success,   # 성공 시에만 알림
) as dag:

    # 1) CRAWL: 이미지 ENTRYPOINT 무시하고, 지정한 스크립트만 실행
    crawl = DockerOperator(
        task_id="crawl",
        image=IMAGE,
        entrypoint="bash",                                 # 👈 ENTRYPOINT 덮어쓰기
        command=["-lc", "python data-prepare/main.py"],    # 👈 내가 원하는 커맨드만 실행
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove=True,
        force_pull=True,
        mount_tmp_dir=False,
        volumes=[SHARED_VOLUME],
        environment={
            "SHARED_DIR": "/opt/shared",
            "RAWG_API_KEY": "{{ var.value.RAWG_API_KEY | default('') }}",
            "RAWG_BASE_URL": "{{ var.value.RAWG_BASE_URL | default('https://api.rawg.io/api/games') }}",
        },
    )

    # 2) TRAIN: PYTHONPATH 추가( src.* 임포트 해결 ), ENTRYPOINT 덮어쓰기
    train = DockerOperator(
        task_id="train",
        image=IMAGE,
        entrypoint="bash",                                 # 👈 ENTRYPOINT 덮어쓰기
        command=["-lc", "python mlops/src/main.py"],       # 👈 우리가 만든 트레이닝 래퍼
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove=True,
        force_pull=True,
        mount_tmp_dir=False,
        volumes=[SHARED_VOLUME],
        environment={
            "SHARED_DIR": "/opt/shared",
            "MODEL_DIR": "/opt/shared/model",
            "GAMES_LOG_PATH": "/opt/shared/games_log.csv",
            "WANDB_API_KEY": "{{ var.value.WANDB_API_KEY | default('') }}",
            "WANDB_MODE": "online",
            "PYTHONPATH": "/opt/mlops",                    # 👈 src.* import 해결
        },
    )

    # 3) INFER: 필요시 PYTHONPATH 추가, ENTRYPOINT 덮어쓰기
    infer = DockerOperator(
        task_id="batch_infer",
        image=IMAGE,
        entrypoint="bash",
        command=["-lc", "python mlops/src/inference/inference.py"],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove=True,
        force_pull=True,
        mount_tmp_dir=False,
        volumes=[SHARED_VOLUME],
        environment={
            "MODEL_DIR": "/opt/shared/model",
            "RECO_PATH": "/opt/shared/recommendations.json",
            "USER_ID": "{{ dag_run.conf.get('user_id', 5) }}",
            "TOP_K": "{{ dag_run.conf.get('top_k', 5) }}",
            "PYTHONPATH": "/opt/mlops",
        },
    )

    crawl >> train >> infer
