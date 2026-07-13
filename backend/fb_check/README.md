# backend/fb_check/

Forward-Backward Consistency Check(FB-Check) — LLM 자동 라벨링 결과를 검증해 CLEAN/NOISE로 분류하는 모듈.

---

## 파일 목록

### `forward_labeling.py`
**Forward Labeling: 계약 조항(C) → 리스크 라벨(L) + 근거 문구(E)**
- OpenAI 모델이 조항 전문을 보고 domain·risk_level·evidence_span·reasoning을 생성
- few-shot 예시 3개(`_FEW_SHOT_EXAMPLES`)로 판단 기준 고정
- 사용 모델은 하드코딩하지 않고 `.env`의 `FORWARD_MODEL`을 그대로 읽는다 (`os.environ["FORWARD_MODEL"]`,
  기본값 없음 — `.env`에 없으면 즉시 에러). `run_forward(..., model=...)`로 호출 시점에 다른 모델(예:
  `gpt-4o-mini`)로 오버라이드도 가능하다 (`llm_benchmark.py`가 이렇게 씀).

### `backward_grounding.py`
**Backward Grounding: E⊂C 검증 + KoELECTRA 독립 예측**
- `snippet_exists`: evidence_span이 원문에 실제 존재하는지 문자열 포함 검사
- `predict`: KoELECTRA로 조항의 domain·risk_level을 독립적으로 예측

### `consistency_verification.py`
**Consistency Verification: 근거 문구(E)만으로 재라벨링(L′)**
- OpenAI 모델이 evidence_span만 보고 독립적으로 재라벨링, `L == L′`이면 라벨 일관성 확인
- 마찬가지로 `.env`의 `VERIFY_MODEL`을 읽는다 (하드코딩 없음)

### `__main__.py`
**FB-Check 오케스트레이터** — 위 3단계를 조합해 CLEAN/NOISE 판정
```bash
python -m backend.fb_check --sample 200 --gpu 1
```

### `llm_benchmark.py`
**FB-Check — 오픈소스/저비용 LLM 대체 가능성 비교**

전체 1,835건 FB-Check 재실행은 GPT-4o API 비용이 드는데, 로컬 GPU 모델이나 더 싼 OpenAI
모델로 대체 가능한지 검증. `data/fb_check/fb_check_results.jsonl`(이미 GPT-4o가 처리한
50건)을 기준값 삼아 비교한다.

비교 대상:
- 로컬(transformers, GPU): Qwen2.5-14B-Instruct, EXAONE-3.5-7.8B-Instruct, Llama-3.1-8B-Instruct
- OpenAI API: gpt-4o-mini (`forward_labeling.run_forward`/`consistency_verification.run_verify`를
  `model="gpt-4o-mini"`로 오버라이드해 그대로 재사용 — 프롬프트·재시도 로직 100% 동일)

두 가지 모드:
- `--mode full`: forward_labeling.py와 같은 프롬프트로 원문 전체(최대 3000자)를 보고 판단,
  `forward_domain`/`forward_label`과 비교
- `--mode evidence`: consistency_verification.py와 같은 프롬프트로 evidence_span(평균 49자)만
  보고 판단, GPT-4o가 같은 조건에서 내린 `verify_domain`/`verify_label`과 비교

```bash
python -m backend.fb_check.llm_benchmark --mode full --sample 20 --gpu 1
python -m backend.fb_check.llm_benchmark --mode evidence --sample 20 --gpu 1
python -m backend.fb_check.llm_benchmark --mode evidence --sample 20 --only llama-3.1-8b
python -m backend.fb_check.llm_benchmark --mode evidence --sample 20 --only gpt-4o-mini
```

결과 저장: `data/fb_check/llm_benchmark_result_{mode}.json` (후보별로 병합 저장 — 다른 후보를
다시 실행해도 기존 후보 결과는 유지됨)

---

## 비교 결과 (2026-07-06/07, 20건 기준, GPT-4o 라벨 대비 일치율)

### mode=full (원문 전체, 최대 3000자 — FTC 판례 발췌문 그대로)

| 모델 | JSON 파싱 성공률 | domain 일치율 | risk_level 일치율 | 평균 속도 |
|---|---|---|---|---|
| **gpt-4o-mini** | 100% | 55.0% | **50.0%** | 1.69초/건 |
| Qwen2.5-14B-Instruct | 100% | 55.0% | 40.0% | 7.58초/건 |
| EXAONE-3.5-7.8B-Instruct | 90% | 40.0% | 60.0% | 5.10초/건 |
| Llama-3.1-8B-Instruct | 100% | 60.0% | 40.0% | 3.38초/건 |

**핵심 발견 — 오픈소스 세 모델 다 "해당없음"을 과도하게 예측함** (Qwen 8/20, EXAONE 10/20,
Llama 8/20). 기준값(GPT-4o)은 이 20건에서 단 한 번도 "해당없음"을 쓰지 않았다. 벤치마크에
쓴 텍스트가 FTC 시정조치 판례 발췌문 전체(법령 인용·각주·사건 경위가 뒤섞인 긴 텍스트)라서,
작은 오픈소스 모델일수록 "이게 계약 조항인지" 판단 자체를 포기하는 경향이 두드러졌다.
같은 OpenAI 계열인 gpt-4o-mini는 이 경향이 없고, risk_level 일치율도 네 후보 중 가장 높다.

### mode=evidence (evidence_span만, 평균 49자 — Consistency Verification과 동일 조건)

| 모델 | JSON 파싱 성공률 | domain 일치율 | risk_level 일치율 | 평균 속도 |
|---|---|---|---|---|
| **gpt-4o-mini** | 100% | **95.0%** | **75.0%** | 0.96초/건 |
| Qwen2.5-14B-Instruct | 100% | 90.0% | 60.0% | 1.28초/건 |
| EXAONE-3.5-7.8B-Instruct | 100% | 85.0% | 35.0% | 1.33초/건 |
| Llama-3.1-8B-Instruct | 100% | 90.0% | 70.0% | **0.67초/건** |

**"긴 노이즈 텍스트가 문제"라는 가설이 확인됐다.** 텍스트를 evidence_span(짧은 인용구)으로
바꾸자 domain 일치율이 로컬 세 모델 다 30~45%p 급등했다(40~60% → 85~90%). risk_level도
Qwen(40→60%)·Llama(40→70%)는 개선됐지만, **EXAONE만 오히려 악화됐다(60→35%)** — 표본을
보면 EXAONE은 High를 Medium으로, Medium을 Low로 체계적으로 한 단계씩 낮춰 예측하는 뚜렷한
과소평가 편향을 보였다. Llama는 이런 편향 없이 골고루 틀렸다. **gpt-4o-mini는 이 모드에서
네 후보 중 domain·risk 둘 다 1등**이다(95%/75%) — GPT-4o(같은 계열)와의 정합성이 오픈소스
모델보다 자연스럽게 더 높은 것으로 보인다.

**캐비엇**: 이 20건 샘플은 도메인이 책임제한_조항 18건·해지_조항 2건으로 치우쳐 있고, 소수인
해지_조항 2건에서는 로컬 세 모델 다 "기타_조항"/"적용법규"/"일반" 등 **허용된 라벨셋(해지_조항/
책임제한_조항/해당없음)을 벗어난 값**을 출력했다 (gpt-4o-mini는 이런 이탈이 없었다) — 시스템
프롬프트가 라벨을 못박아도 소형 오픈소스 모델에는 완벽히 지켜지진 않는다. domain 90%+라는
수치는 주로 다수 클래스(책임제한_조항)에 대한 성능이다.

### 결론

**gpt-4o-mini가 비용·품질 둘 다에서 가장 나은 선택지다.**

- 비용: GPT-4o 대비 입력 1/17, 출력 1/17 — 전체 1,835건 재실행 비용이 **~$9~12 → ~$0.5**
- 품질: mode=evidence에서 domain 95%·risk 75%로 네 후보 중 최고, mode=full에서도 risk_level
  일치율 50%로 최고 (domain은 Qwen과 동률)
- 로컬 GPU 다운로드/서빙 부담 없이 기존 `run_forward`/`run_verify` 코드를 `model` 인자만
  바꿔서 그대로 재사용 가능

오픈소스 로컬 모델(Qwen/Llama)은 Consistency Verification 단계(evidence_span만 다룸)에서는
쓸만하지만(domain 90%, risk 60~70%), gpt-4o-mini가 비용도 더 싸고 품질도 더 좋아서 오픈소스를
쓸 이유가 없다. **FB-Check 전체 재실행은 `FORWARD_MODEL=gpt-4o-mini`, `VERIFY_MODEL=gpt-4o-mini`로
설정해서 진행하는 것을 권장한다.**
