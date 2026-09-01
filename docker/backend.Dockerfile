# docker/backend.Dockerfile
#
# ContractRiskGuard FastAPI 서버 이미지. 빌드 컨텍스트는 저장소 루트(..)여야
# 한다 — backend/ 코드와 requirements.txt를 함께 COPY하기 위해서다.
# (docker-compose.yml에서 build.context: .. 로 지정)
#
# requirements.txt가 곧 "서버가 실제로 기동 가능한 최소 세트"라 별도
# requirements-serve.txt 없이 이 파일을 그대로 쓴다.
#
# **torch는 기본적으로 CPU wheel을 받는다.** 이유는 이미지 크기가 아니라 정합성이다 —
# docker-compose.yml의 GPU 블록(deploy.resources)이 기본 주석 처리라, 이 이미지는
# 그대로 띄우면 **어차피 GPU를 못 쓴다.** 그런데 예전 기본값은 CUDA wheel이라
# nvidia-* 런타임 5.7GB를 받아놓고 한 번도 안 썼다(빌드가 디스크·시간에서 죽기도 한다).
# `resolve_embed_device()`도 GPU가 없으면 cpu로 내려가므로 동작은 같다.
#
# **GPU로 띄우려면 셋을 같이 바꿔야 한다** — 하나만 바꾸면 조용히 CPU로 돈다:
#   1) 빌드:  --build-arg TORCH_INDEX=https://pypi.org/simple
#   2) 실행:  docker run --gpus all  (또는 compose의 deploy 블록 주석 해제)
#   3) 호스트: nvidia-container-toolkit 설치
# CPU wheel은 CUDA 커널이 아예 빠져 있어 섞어 쓸 수 없다.
#
# models/, data/는 크고 git에 없는 아티팩트라 이미지에 굽지 않고 런타임에
# 볼륨으로 마운트한다(docker-compose.yml의 api 서비스 volumes 참고).

FROM python:3.12-slim

WORKDIR /app

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# **torch도 핀한다.** 락에서 뺐다고 안 고정하면, 이미지에서 제일 크고 동작에 제일
# 중요한 패키지 하나만 매 빌드 최신으로 떠서 락을 뜬 의미가 반감된다.
# `+cpu` 로컬 태그는 빼고 쓴다 — 위 인덱스에서 `2.13.0`이 곧 `2.13.0+cpu`로 풀린다
# (그래서 TORCH_INDEX를 PyPI로 바꾸면 같은 버전의 CUDA 빌드가 온다).
ARG TORCH_VERSION=2.13.0

# **이미지는 `requirements-lock.txt`로 짓는다** — `requirements.txt`가 아니다.
#
#   requirements.txt        사람이 쓰는 **의도**. `>=` 하한만 있다
#   requirements-lock.txt   기계가 만든 **실제로 도는 조합**. `pip freeze` 산출물
#
# 하한만 있으면 빌드할 때마다 최신 메이저로 올라간다. 2026-09-01 빌드에서 호스트와
# openai 2.36→3.6 · pandas 2.3→3.0 · sentence-transformers 5.4→6.0 만큼 벌어졌다.
# 그날은 통과했지만 **다음 빌드가 같은 이미지라는 보장이 없었다.**
#
# 락은 손으로 고치지 말 것. 버전을 올리려면 `requirements.txt`를 고치고 락을 다시 뜬다:
#   sudo docker run --rm crg-api:test pip freeze | grep -v '^torch==' > requirements-lock.txt
#
# torch를 락에서 빼는 이유: `pip freeze`가 `torch==2.x.x+cpu`로 내는데 그 **로컬 버전
# 태그는 기본 인덱스에서 해석되지 않는다.** torch는 아래처럼 전용 인덱스로 따로 깐다.
COPY requirements-lock.txt .
# torch를 **먼저** 지정한 인덱스에서 받는다. `--extra-index-url`로 섞으면 pip이
# PyPI의 CUDA wheel을 골라버려 의도가 조용히 뒤집힌다 — 그래서 `--index-url`로
# 분리 설치하고, 두 번째 pip은 이미 만족된 torch를 건드리지 않는다.
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch==${TORCH_VERSION} \
 && pip install --no-cache-dir -r requirements-lock.txt

COPY backend/ backend/

EXPOSE 8000

CMD ["uvicorn", "backend.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
