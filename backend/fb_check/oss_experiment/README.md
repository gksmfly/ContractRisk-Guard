# backend/fb_check/oss_experiment/

FB-Check 오픈소스 LLM **대조군 실험** — `backend/fb_check/llm_benchmark.py`의 20건 소규모 비교에서
Qwen2.5-14B-Instruct와 Llama-3.1-8B-Instruct가 evidence_span 기준으로는 쓸만한 일치율(85~90%
domain, 60~70% risk)을 보였는데, 이게 소규모 샘플의 우연인지 전체 데이터(1,835건)에서도
유지되는지 실제로 돌려서 확인하는 실험이다.

`backend/fb_check/__main__.py`(GPT-4o 버전)와 동일한 3단계 파이프라인을 그대로 쓰되, Forward
Labeling과 Consistency Verification 두 GPT-4o 호출만 로컬 오픈소스 모델로 교체한다.
Backward Grounding(KoELECTRA)은 완전히 동일한 코드를 재사용한다.

---

## 파일 목록

### `local_labeling.py`
`forward_labeling.run_forward()` / `consistency_verification.run_verify()`의 로컬 모델
버전. 같은 시스템 프롬프트·few-shot 예시(`_SYSTEM`/`_FEW_SHOT_EXAMPLES`)를 그대로 가져다
쓰고, OpenAI API 호출 대신 `transformers`로 로컬 GPU 추론을 한다.

### `__main__.py`
오케스트레이터. `backend/fb_check/__main__.py`와 로직은 동일(Forward → Backward → Verify →
CLEAN/NOISE 판정)하고, Forward/Verify만 `--model`로 지정한 로컬 모델을 쓴다.

```bash
# 전체 1,835건
python -m backend.fb_check.oss_experiment --model qwen2.5-14b --gpu 0
python -m backend.fb_check.oss_experiment --model llama-3.1-8b --gpu 1

# 스모크 테스트
python -m backend.fb_check.oss_experiment --model llama-3.1-8b --sample 5 --gpu 1
```

배치마다(`--save-every`, 기본 50건) 중간 저장하고, `chunk_id` 기준 체크포인트를 남겨서
중단 후 재실행하면 이어서 처리한다.

## 출력

```
data/fb_check/oss_experiment/
├── qwen2.5-14b/
│   ├── fb_check_results.jsonl
│   ├── clean.jsonl
│   ├── noise.jsonl
│   └── fb_check_report.json
└── llama-3.1-8b/
    └── (동일 구조)
```

`data/fb_check/{clean,noise,fb_check_report}.json`(GPT-4o 버전, 최상위)과 겹치지 않게
모델별 하위 디렉터리로 분리했다.

## 예상 소요 시간 (1,835건 기준, `llm_benchmark.py` mode=full 속도 실측 기준)

| 모델 | Forward(7.59s/건 또는 3.38s/건) | Verify(~60% 도달, 1.28s/건 또는 0.67s/건) | 합계 |
|---|---|---|---|
| Qwen2.5-14B | ~3.9시간 | ~23분 | **~4.2시간** |
| Llama-3.1-8B | ~1.7시간 | ~12분 | **~1.9시간** |

GPU가 2장(cuda:0, cuda:1)이라 두 모델을 동시에 각각 다른 GPU에서 돌리면 총 대기 시간은
더 느린 쪽(Qwen, ~4.2시간)에 맞춰진다.

---

## 결과

(실행 후 이 섹션에 GPT-4o 버전 `data/fb_check/fb_check_report.json`과의 CLEAN/NOISE 비율,
라벨 분포 비교표를 채운다.)
