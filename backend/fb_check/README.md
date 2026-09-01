# backend/fb_check/

Forward-Backward Consistency Check(FB-Check) — LLM 자동 라벨링 결과를 검증해 CLEAN/NOISE로 분류하는 모듈.

---

## ⚠️ 라벨링 재개 절차 — 먼저 읽을 것 (2026-09-01)

라벨링은 **52.3%에서 멈춰 있다.** 08-23에 OpenAI 크레딧이 소진돼 중단됐고, 그 상태 그대로다.

    총 입력    4,466건
    확보       2,335건 (52.3%)   CLEAN 1,786 / NOISE 549
    미처리     2,131건 (47.7%)   status=ERROR로 기록됨 → 재개 시 자동으로 재시도 대상

### 반드시 `--model gpt-4o`를 붙일 것

**기존 2,335건은 전량 `gpt-4o`로 만들어졌다**(레코드의 `forward_model`/`verify_model` 필드로
확인 가능). 그런데 `.env`는 `FORWARD_MODEL=gpt-4o-mini`다 — 아래 "결론" 절의 07-06 권고가
그대로 남아 있어서인데, **그 권고는 실행되지 않았다.**

그냥 재개하면 [`_load_checkpoint`](__main__.py)의 stale 가드가 작동해서 **기존 레코드를
"모델 불일치 → 재처리 대상"으로 잡는다.** 조용한 오염(gpt-4o 라벨과 mini 라벨이 한 파일에
섞이는 것)은 구조적으로 막히지만, 대신 처리 대상이 두 배가 된다.

**09-01 `--dry-scope` 실측 — 플래그 하나로 $17.5 차이가 난다:**

| 명령 | 경고 | 처리 대상 | 예상 비용 |
|---|---|---|---|
| `--model gpt-4o --dry-scope` | 없음 | **2,131건** | **$15.98** |
| `--dry-scope` (`.env`=mini) | `모델·프롬프트가 다른 레코드 2482건 → 재처리 대상` | 4,468건 | $33.51 |

```bash
# ① 먼저 범위를 확인한다 — API를 한 번도 호출하지 않고, GPU도 안 쓴다
.venv/bin/python -m backend.fb_check --model gpt-4o --dry-scope

#    기대 출력:  ★ forward: gpt-4o   ★ verify : gpt-4o
#               처리 대상  2131건    예상 비용 $15.98
#    "모델·프롬프트가 다른 레코드 N건" 경고가 **한 줄이라도 뜨면 실행하지 말 것** —
#    --model을 빠뜨렸거나 프롬프트 버전이 바뀐 것이다.

# ② 범위가 맞으면 실행
.venv/bin/python -m backend.fb_check --model gpt-4o --gpu 1
```

`.env`를 `FORWARD_MODEL=gpt-4o`로 바꿔도 결과는 같다. 다만 `.env`는 실험 스크립트도 함께
읽으므로(`label_pilot`이 gpt-4o, `fb_check`이 mini를 쓰던 것이 정확히 이 문제였다),
**어느 쪽을 고르든 재개 전에 `--dry-scope`로 확인하는 절차 자체는 건너뛰지 말 것.**

`--dry-scope`를 건너뛰지 말 것 — 08-24에 `--redo-reason snippet_not_found`가 의도한 146건이
아니라 2,277건을 대상으로 잡아 $1짜리 작업이 $17이 될 뻔한 적이 있다
(`_redo_ids` docstring 참고).

### 재개 후 반드시 다시 돌려야 하는 것

라벨이 늘면 아래 수치가 전부 움직인다. 라벨 구성만 바뀐 08-25 재처리(148건 재실행, 86건이
NOISE→CLEAN)만으로도 출처 교락이 +0.9%p → +2.4%p로 2.7배가 됐다.

```bash
.venv/bin/python -m backend.eval.confound_articles      # 출처 교락 (학습 전 필수)
.venv/bin/python -m backend.eval.noise_reason_audit      # FB-Check 게이트가 값을 하는지
.venv/bin/python -m backend.training.train_article --gpu 1 --num-workers 0
```

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
쓸 이유가 없다. ~~**FB-Check 전체 재실행은 `FORWARD_MODEL=gpt-4o-mini`, `VERIFY_MODEL=gpt-4o-mini`로
설정해서 진행하는 것을 권장한다.**~~

> **[정정 2026-09-01] 이 권고는 따르지 말 것 — `.env`가 mini인 이유가 여기다.**
>
> 이 권고는 `.env`에 반영됐지만 **실제 라벨링은 gpt-4o로 실행됐다**(현재 2,335건 전량의
> `forward_model`/`verify_model`이 `gpt-4o`). 즉 `.env`와 산출물이 갈린 채로 남아 있고,
> 그대로 재개하면 위 "라벨링 재개 절차"의 두 배 비용 문제가 그대로 터진다.
>
> 권고 자체의 근거도 지금은 약하다:
> - **표본 20건**이고, 비교 기준이 "GPT-4o와 얼마나 일치하는가"이지 외부 정답 대비
>   정확도가 아니다. 저 표는 mini가 옳다는 근거가 아니라 mini가 4o를 닮았다는 근거다.
> - **2-도메인 taxonomy 시절**(2026-07-06) 측정이다. 지금 라벨은 §6~14 multi-label이라
>   과제 자체가 다르다.
> - 표본이 책임제한 18 / 해지 2로 치우쳐 있다고 같은 문단이 이미 밝히고 있다.
>
> **판단**: 이미 절반이 gpt-4o로 만들어졌으므로 **일관성을 위해 gpt-4o로 마저 돌린다.**
> 비용을 아끼려고 지금 mini로 갈아타면 한 학습 세트 안에 두 판정자가 섞여, 앞으로 나오는
> 모든 수치에서 "모델 차이인지 데이터 차이인지" 영영 못 가린다. mini 전환을 진짜로
> 검토하려면 §6~14 taxonomy에서 **FTC 근거_법령 대비** 다시 재고, 그때는 라벨을
> 새로 다 만드는 것이 전제다.

---

## 정량 평가 — FTC 확정 판정 대비 실제 성능 (2026-07-13)

지금까지의 모든 수치(검증 F1 93.9% 등)는 **GPT가 자기 자신과 일관되는지**를 잰 것이지,
**실제로 법적으로 맞는 판단인지**를 잰 게 아니었다 — CLEAN 판정 자체가 `forward_label ==
verify_label`(GPT vs GPT)만 비교하고, `seed_risk`(원본 정답)는 비교에 안 쓰인다.

그런데 `seed_labeled.jsonl`의 `risk_level`은 사실 GPT가 만든 게 아니다
(`backend/labeling/seed.py` 참고):
- **FTC 시정조치 사례** → `risk_level=High` **하드코딩** (공정거래위원회가 이미 공식적으로
  위반 판정을 내린 사건 — GPT의 추측이 아니라 정부 기관의 실제 법적 판단)
- **표준계약서** → 정규식 패턴 기반 (GPT 미사용)

즉 `seed_risk`는 GPT와 독립적인 정답으로 쓸 수 있다. FTC 시정조치 499건(전부 확정 High)을
기준으로 실제 성능을 측정했다.

| | High 정답과 일치 |
|---|---|
| GPT-4o-mini (forward_label) | 241/499 (48.3%) |
| KoELECTRA v2 (evidence_span 있음, 304건) | 183/304 (**60.2%**) |
| KoELECTRA v2 (evidence_span 없음→원문 대체, 195건) | 23/195 (11.8%) |
| KoELECTRA v2 (전체 499건 블렌드) | 206/499 (41.3%) |

**핵심 결론**:
1. v2 자체는 학습 때와 같은 입력 형태(evidence_span)를 받으면 60.2% — 3지선다 랜덤(33%)보다는
   낫지만 자기일관성 지표(93.9%)와는 거리가 멀다.
2. 더 심각한 문제: FTC 확정 위반 499건 중 **195건(39%)은 GPT-4o-mini가 Forward Labeling
   단계에서 evidence_span도 못 뽑고 "해당없음"으로 처리**했다 — KoELECTRA까지 가지도 못하고
   그 앞에서 누락된다.
3. 이 195건을 직접 읽어보니 상당수가 실제로는 **약관규제법 조문 자체나 심결 절차 서술문이
   조항 원문으로 잘못 추출**된 것이었다 — `backend/scripts/parse_ftc_case_pdf.py`의 정규식이
   너무 느슨해서, 피심인의 실제 약관뿐 아니라 법 조문·증거 인용까지 "조항 원문"으로 섞어
   추출하고 있었다(자세한 내용과 수정은 `backend/scripts/README.md` 참고).
4. **즉 39% 누락의 상당 부분은 GPT나 KoELECTRA의 판단 문제가 아니라, 그보다 훨씬 앞
   단계(Step 0 PDF 파싱)의 데이터 품질 문제였다.**

**주의**: 파싱 로직은 수정했지만(`backend/scripts/parse_ftc_case_pdf.py`), 이미 만들어진
`seed_labeled.jsonl`/FB-Check 결과/KoELECTRA v2는 이 수정 이전 데이터로 만들어진 것이라
아직 반영 안 됐다. 실제로 성능이 얼마나 개선되는지 확인하려면 PDF 재파싱 → seed 재라벨링 →
FB-Check 재실행 → 재학습까지 전체 파이프라인을 다시 돌려야 한다.
