# docker/backend.Dockerfile
#
# ContractRiskGuard FastAPI 서버 이미지. 빌드 컨텍스트는 저장소 루트(..)여야
# 한다 — backend/ 코드와 requirements.txt를 함께 COPY하기 위해서다.
# (docker-compose.yml에서 build.context: .. 로 지정)
#
# requirements.txt가 곧 "서버가 실제로 기동 가능한 최소 세트"라 별도
# requirements-serve.txt 없이 이 파일을 그대로 쓴다.
#
# KoE5 임베더는 기본적으로 GPU(EMBED_DEVICE, 기본값 cuda:1)를 쓴다. 아래는 GPU용
# CUDA 지원 torch 기본 wheel을 그대로 설치한다(용량이 크다) — 호스트에
# nvidia-container-toolkit이 설치돼 있어야 하고, `docker run --gpus all`
# (또는 compose의 deploy.resources.reservations.devices)로 실행해야 실제로 GPU를
# 쓴다. GPU 없는 호스트에 배포한다면 .env에 EMBED_DEVICE=cpu로 설정하고, 이미지도
# 가볍게 하려면 아래 pip install에 `--extra-index-url
# https://download.pytorch.org/whl/cpu`를 추가해 CPU 전용 wheel을 받도록 바꿀 것
# (CPU wheel은 CUDA 커널이 아예 빠져있어 GPU를 절대 못 쓴다 — 섞어 쓸 수 없음).
#
# models/, data/는 크고 git에 없는 아티팩트라 이미지에 굽지 않고 런타임에
# 볼륨으로 마운트한다(docker-compose.yml의 api 서비스 volumes 참고).

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/

EXPOSE 8000

CMD ["uvicorn", "backend.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
