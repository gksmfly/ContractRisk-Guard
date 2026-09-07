# ContractRisk-Guard: Contract Clause Risk Prediction via Forward-Backward Consistency Check

> **TL;DR**: Predicts which articles of the Korean Unfair Terms Regulation Act (약관규제법 §6–§14)
> a contract clause violates. Training labels are produced without human annotators by a 3-stage
> FB-Check pipeline (GPT-4o labels → evidence grounding → consistency re-labeling), and a fine-tuned
> KoELECTRA does the inference. Published at KAICTS 2025.
>
> **This README was substantially revised on 2026-08-31.** Follow-up measurement retracted two claims
> from the published paper and validated a third. See [Revisions](#revisions-2026-08-31).

---

## Problem Statement

Korean service contracts frequently contain clauses that are unfair under the Unfair Terms Regulation
Act (약관규제법) — unilateral termination (§9), liability limitation (§7), one-sided performance terms
(§10), and six other article-level categories — yet are too dense for non-experts to review manually.

- Existing LLM-based legal systems lack a verification layer for generated judgments
- LLM auto-labeling creates self-consistency bias: errors accumulate when the same model family validates its own outputs
- Legal-domain mislabeled data compounds through retraining, degrading both reliability and explainability

---

## Approach

- **FB-Check to filter hallucinations from auto-labeled data**: Forward Labeling (GPT-4o generates label L and evidence span E from clause C) → Backward Grounding (1st-generation KoELECTRA validates E ⊂ C at index level) → Consistency Verification (re-labels from E alone; L′ == L → CLEAN, otherwise NOISE)
- **Data Flywheel** — *suspended after three attempts, none of which showed an effect.* The most
  recent (2026-09-06) put `article_v2` in as a third voter and first reported +36.7%p; that was
  **withdrawn within the day** once it turned out the exclusion rule swept in every record whose GPT
  label was empty — those score 0 against a reference that never is, so the rule was manufacturing
  its own signal. Re-run symmetrically, all three exposure layers are undetermined, and the layer
  with the most power (685 vs 38) puts the effect size at **d = −0.01**. Worth stating plainly what
  even a positive result would have bought: this measures whether a third voter vetoes bad labels,
  which is a necessary condition for the paper's claim — that repeated retraining jointly optimises
  label quality and generalisation — and not the claim itself.
- **KoELECTRA as the inference engine**: The fine-tuned classifier makes all risk judgments; GPT-4o is strictly limited to explanation generation and search query construction — reducing LLM calls, latency, and cost while keeping judgments consistent
- **A fixed 6-stage pipeline** (not agentic): Analysis → Retrieval Strategy → Evidence Selection →
  Judgment (fine-tuned KoELECTRA) → Red-team → Evidence Verification. **No LLM controls the flow.**
  The Retrieval Strategy stage originally used a local LLM to pick which statutes to search; on a
  100-clause head-to-head it won **0 times** against a constant that always searches 약관규제법
  §6–§14, and law-retrieval hit rate went **18% → 81%** once the router was replaced by that constant.
  Calling this "multi-agent collaboration" overstated it, so the term is dropped.
- **Article-level taxonomy (§6–§14), not two domains**: the original 2-domain scheme (termination /
  liability limitation) discarded **56.7% of pipeline input** (1,257 of 2,218 clauses) as "not
  applicable" — most of which were real violations of articles the scheme had no slot for.
- **FTC enforcement cases as seed data**: 공정거래위원회 시정조치 cases are objectively High-risk
  (public enforcement = confirmed violation), avoiding annotator subjectivity

---

## Key Results

Scored against an external reference the pipeline never sees: the articles the Korean FTC actually
cited (`근거_법령`) in enforcement decisions. All intervals are 95% paired bootstrap CIs.

**Population.** Row 1 is scored only on records where the labeler named at least one article; records
with an empty prediction are excluded from *both* groups. FTC gold is never empty, so an empty
prediction scores 0 by construction, and the two groups differ sharply in how often they are empty
(CLEAN 21.4%, NOISE 2.2%). Without this condition the comparison measures that composition difference
rather than label quality, and the sign reverses: **−0.3%p [−4.0, +3.3], not significant.** Both
readings are legitimate but answer different questions — *"would adding these records lower the
training-set average?"* (no) versus *"is what these records assert as trustworthy as what CLEAN
asserts?"* (no). A label filter should be judged on the second.

Rows 3–5 use a different population: the 255-case clean evaluation stratum (single clause **and**
single cited article), with **no** filter on predictions — an empty prediction there simply scores 0,
as it should. Row 2 uses all CLEAN records, empty predictions included, because an empty label is a
legitimate outcome for the standard-contract half of that corpus.

| Question | Measurement |
|---|---|
| Does FB-Check select better labels? | **+3.8%p** [+1.1, +6.6] — the `agreed` label that actually enters the training set (CLEAN 42.8%, n=1,150) vs the `forward` label that was discarded (NOISE 39.0%, n=816), article-F1 against FTC-cited articles, non-empty predictions, MW p=1.5e-03. Scoring both sides with the `forward` label (the filter's raw selectivity) gives +7.7%p [+5.1, +10.3]; the headline uses the smaller, claim-matching number. On the ambiguity-free stratum (single clause, single cited article) the gap is +6.9%p [−4.6, +18.5] — **undetermined**; 80% power needs ~601 NOISE records and there are 92. By discard reason: article_mismatch +11.3%p [+4.9, +17.3], label_mismatch +6.9%p [+4.2, +9.6] — but only in one direction: forward *Medium* → verify *High* is +10.9%p worse (n=354), while forward *High* → verify *Medium* (n=189) is indistinguishable from CLEAN (−2.6%p [−6.4, +1.2]). Script: `backend/eval/fbcheck_external_eval.py`; report: `data/eval/fbcheck_external_report.json`. The earlier +8.8%p [+5.1, +12.7] (630/395 snapshot, README-only) is superseded |
| Does the model read the clause, or the corpus it came from? | Source-conditional constant beats an unconditional one by only **+2.4%p** — the source shortcut is small (was +0.9%p on the 1,700-record snapshot; re-measured on all 1,786) |
| Does the model beat "always guess the three most common articles"? | **Yes on the scoreable stratum, at cross-fitted thresholds**: teacher (GPT-4o) +9.0%p [+2.9, +15.4], student (110M KoELECTRA) +6.4%p [+1.6, +11.4]. **At the deployed thresholds the student is +2.5%p [−3.7, +9.2] — undetermined.** The gap between the two is the calibration deficit |
| Does a larger model help? | **Not detectable at this resolution.** GPT-4o and the 110M student differ by −2.4%p [−6.4, +1.7] — an interval that contains zero. This is *underpowered*, not *no effect* |
| Does more training data help? | **Below the pre-registered threshold.** 300 → 900 moves gold F1 by −2.1%p (threshold: +7%p); 1,329 → 2,906 gives +0.9%p [−3.5, +5.4]. Detecting 3%p needs n≈3,400 and gold has 255 — deprioritised, not closed |

**The evaluation set itself was the main confound.** 22% of it (72 of 327 cases) attributed a
*case-level* list of cited articles to a *single* clause, because the PDF parser had found only one
clause where the decision described on average 2.89 violations — **72 of 72, without exception**.
On that stratum the optimal constant is "always predict four articles" (F1 56.6%), which no model
reading a single clause can match. Mixing it with the clean 78% is what made every aggregate read
as "no significant difference".

| Corpus | Size |
|---|---|
| Legal corpus | 16 laws, 26 interpretations, 1,995 court precedents, 2,488+ FTC cases |
| Auto-labeled clauses (no human annotators) | 2,335 scored / 1,786 CLEAN + 549 NOISE (52.3% of the 4,466 seed records; the rest is unprocessed, not discarded) |
| External evaluation set | 255 FTC cases (single clause **and** single cited article) |
| Published | KAICTS 2025 |

<details>
<summary>Superseded figures from the original paper</summary>

`CLEAN label match rate 94.9% (131/138)` and `NOISE label match rate 76.2% (439/576)` were computed
under the 2-domain taxonomy and measure agreement between two GPT-4o passes — not agreement with any
external reference. They are not comparable to the table above and should not be cited alongside it.

</details>

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python, uvicorn |
| Core Inference | KoELECTRA fine-tuned on CLEAN data — article multi-label head (§6–§14, sigmoid + BCE), served from `models/article_v2`. The superseded dual-head (domain + risk) model is retained only for the labeling path |
| LLM (support) | OpenAI GPT-4o (explanation generation, search query construction only) |
| Retrieval | Dense + Sparse Hybrid Retrieval, metadata filtering |
| Frontend | Next.js 14, Tailwind CSS, shadcn/ui, Zustand |
| Database | PostgreSQL + pgvector (Docker) |
| PDF Processing | pdfplumber, pdfjs-dist |

---

## Project Structure

```
ContractRisk-Guard/
├── backend/
│   ├── api/                        # FastAPI server, routers, schemas
│   ├── fb_check/                   # FB-Check auto-labeling pipeline
│   │   ├── forward_labeling.py     # GPT-4o: label L + evidence E from clause C
│   │   ├── backward_grounding.py   # KoELECTRA: validate E ⊂ C + independent prediction
│   │   └── consistency_verification.py  # Re-label from E only; L′ == L → CLEAN
│   ├── domain/                     # Legal document filtering
│   ├── preprocess/                 # Text cleaning and chunking
│   ├── labeling/
│   │   ├── articles.py             # §6–§14 taxonomy, generated from the raw statute JSON
│   │   └── seed.py                 # FTC cases + standard contracts → seed data
│   ├── training/
│   │   ├── train.py                # dual-head (superseded, still used by the serving path)
│   │   └── train_article.py        # article multi-label, document-level splits, dev-only thresholds
│   ├── model/electra.py            # DualHeadElectra + ArticleMultiLabelElectra (weight fingerprint)
│   ├── eval/                       # measurement harness — constant baselines, gold strata,
│   │                               # threshold regimes, source-confound, teacher-vs-student
│   └── scripts/                    # Law API crawler, FTC case scraper (caches extracted PDF text)
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # Landing page
│   │   └── analyze/                # Contract analyzer UI
│   └── components/
└── docker/docker-compose.yml       # PostgreSQL pgvector
```

---

## Getting Started

All backend commands run from the **repository root** — the code uses absolute imports
(`from backend.api...`), so running from inside `backend/` fails with `ModuleNotFoundError`.

```bash
# Backend (from the repository root, not backend/)
pip install -r requirements.txt
cp .env.example .env
# Fill in at minimum: OPENAI_API_KEY, DATABASE_URL, FORWARD_MODEL
#   FORWARD_MODEL is required by the *server*, not just the labeling pipeline —
#   the Analysis stage reuses run_forward(). Without it the server refuses to start
#   and names the missing variable.

# Postgres + pgvector
docker compose -f docker/docker-compose.yml up -d postgres

uvicorn backend.api.server:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## Revisions (2026-08-31)

The published paper listed quantitative evaluation as future work. That evaluation has now been
carried out, and it changed three claims. Full measurement logs, including the pre-registered
decision rules and the analyses that failed, are in
[`backend/eval/prompt_block_ablation.md`](backend/eval/prompt_block_ablation.md).

| Claim | Status | Basis |
|---|---|---|
| Data Flywheel | **Suspended — three attempts, none showing an effect** | **2026-08-16** — rejected under the 2-domain taxonomy: a model trained on pipeline output acted as a near-constant voter (+6.1%p over a constant) and, used as a veto, *worsened* source–label confounding (75.5% → 96.9%). **2026-09-01** — downgraded to *suspended*: that mechanism no longer recurs (the source shortcut is now +2.4%p), but the benefit was unmeasurable at n=114. **2026-09-06** — re-measured with `article_v2` as the voter. A first pass reported +36.7%p [+23.2, +49.5] and was **withdrawn the same day**: the exclusion rule auto-included empty GPT labels (30 of 55 excluded records), which score 0 against the never-empty FTC reference — *the apparent signal was the rule*. With a symmetric rule and a three-way split all three layers are undetermined; in the best-powered layer (685 vs 38) the effect size is **d = −0.01**, upper bound roughly +9%p. **Note what this measures**: whether a third voter vetoes bad labels — a *necessary condition*. The paper's claim is that repeated retraining jointly optimises label quality and generalisation, which no selection signal alone would establish |
| 6-agent multi-agent system | **Restated** as a fixed 6-stage pipeline | LLM routing lost 0–100 to a constant; 18% → 81% retrieval after removal |
| FB-Check filters hallucinated labels | **Validated externally** | +3.8%p [+1.1, +6.6] against FTC-cited articles on the label that enters training (`agreed`), non-empty predictions (n=1,150 / 816); +7.7%p [+5.1, +10.3] on the raw `forward` label. Undetermined on the ambiguity-free stratum (n=92 NOISE, ~601 needed). First external check of this claim; reproduced independently by two implementations to the decimal, then fixed in `backend/eval/fbcheck_external_eval.py` |
| 2-domain taxonomy | **Replaced** by §6–§14 | The old scheme discarded 56.7% of input |
| Evaluation set | **Corrected** | 22% was unscoreable by construction (72/72 diagnosed) |

## Limitations & Future Work

- **Deployment thresholds are uncalibrated.** With thresholds fitted to a distribution matching the
  evaluation set, the model beats the constant baseline (+6.2%p [+0.9, +11.7]); with the thresholds
  actually shipped, it does not (+2.5%p [−3.7, +9.2]). The gap is calibration, not capability. The
  training split is 67% empty-label and the evaluation set is 0%, so neither is a usable calibration
  source. Decision-document before/after tables were surveyed as a third option and **do not exist at
  usable scale** (~12 cases corpus-wide, 0 of which meet the single-cited-article condition).
- **§11 and §10 are not learned** (recall 4–11% at n=47 / n=28), and this is neither a data-volume
  nor a threshold effect.
- **Serving migrated to the article multi-label model** (`models/article_v1`, 2026-08-31), and the
  gate moved from GPT's 2-domain value to the model's own article output. Two consequences follow
  from the calibration gap above rather than being fixed by the migration:
  - **No risk tier is shown.** The article head has no risk output (gold was undefinable), and the
    confidence bands were measured on `models/v4` only, so neither could be carried over. The UI
    reports a binary "needs review" plus the predicted articles **as a reference, not a verdict** —
    clause-level recall is **78.4%** against the FTC-cited articles (`article_v2`, max_len 512),
    while article-level is indistinguishable from a constant (38.5% vs 36.1%, CI [−3.7, +9.2] on
    the clean stratum, n=255). The earlier "38.2% vs 40.3%, CI [−7.7, +3.4]" in this file was measured on the old
    327-record definition, 22% of which is unscoreable, and is superseded.
    The companion "2.6% false-alarm" figure was **withdrawn on 2026-09-01**: the negative pool's
    ground truth *is* the GPT label (`forward ∩ verify`), so a clause the model flags and GPT did not
    is counted as a false alarm even when the model is right. It measures **disagreement with GPT**,
    not false alarms. No independently-judged negative set exists yet — see Limitations.
  - **The serving input was not the input the model was trained or scored on** (found and fixed
    2026-09-01). `judgment_node` fed `evidence_span or clause` — a rule inherited from `models/v4`,
    which was trained with span augmentation — while `article_v1` is trained and scored on the full
    clause. Paired on the 136 gold clauses that have a span, feeding the ~43-character fragment
    instead of the ~183-character clause costs **9.6 points of clause-level recall**
    (81.6% → 72.1%, CI [−16.9, −2.9]). Worse, the input is *correlated with the label*: a span
    exists for 53% of violating clauses but only 2% of non-violating ones, so recall and
    disagreement move in opposite directions (78.0% → 72.9% and 2.6% → 4.0%). The reported 78.0%
    described an input production never used. Serving now passes the full clause; a regression test
    pins it (`tests/agents/test_judgment_agent.py`), and `backend/eval/input_parity_eval.py` holds
    the measurement.
  - **`max_len=256` was silently truncating a third of real input** (found and fixed 2026-09-01,
    now `models/article_v2`). It is a training hyperparameter, but in production it acted as data
    loss: real contract clauses bundle several sub-items and run 2.4x longer than the training text,
    so **35.4% lost their tail** — against 5.5% of the gold set, which is why the reported 78.0%
    never showed it. Retraining at 512 was adopted under a **pre-registered no-harm rule** (gold
    clause-level recall 78.0% -> 78.4%, CI [-3.9, +4.7]); the rationale is *not* that 512 scores
    better — picking on gold is forbidden — but that the user's input should not be discarded.
    Serving now reads `max_len` from the checkpoint instead of hardcoding it.
    **Two caveats stand.** Flagging on the 99 real clauses fell 39.4% -> 28.3% (paired -11.1%p,
    CI [-20.2, -2.0]) and nothing available can yet grade whether that is v1 over-flagging on a
    truncated head or v2 diluting a violation in a long clause; the gold truncated subgroup shows
    **0/18 detections lost** (rule of three: at most 16.7%), which argues against the latter. And
    that subgroup only covers mild truncation — gold's worst case loses 25.4% of its tokens while
    **60% of the real truncated clauses lose more than that**, and 8 still exceed 512.
  - **Thresholds are still the dev-split values**, not deployment-calibrated. Re-optimising needs the
    deployment prevalence *r*, which is unmeasured; a 99-clause blind worksheet
    (`backend/eval/prevalence_worksheet.py`) is built and awaiting human labels.
  - **The false-alarm rate has never been measured against an independent reference.** Every
    "correctly stays silent" number in this project is scored against pipeline-produced labels.
    Comparisons *between* configurations remain valid — they all use the same ruler, and that is how
    the `--negative-ratio` arms were ranked — but no absolute claim about false alarms can be made.
    Judging ~50 of the 151 held-out standard-contract clauses by hand would produce the first
    independent estimate.
- **The Red-team stage is inert.** Its neighbour table (`clean_clauses`) still holds the old
  risk-level labels, so no article comparison can fire. It stays silent rather than warning on a
  stale axis; reloading that table with the current `clean.jsonl` revives it.
- **Labeling is 52.3% complete** (2,335 of 4,466; stopped on API credit exhaustion). Resuming
  requires `--model gpt-4o` — the `.env` default is `gpt-4o-mini`, which would flag every existing
  record as stale and double the cost. See `backend/fb_check/README.md`.
- Coverage gaps remain for clauses involving recent law amendments or sparse precedent domains

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Seoyeon Kim** | Undergraduate Researcher, Korean Bible University  
[GitHub](https://github.com/gksmfly) · [Email](mailto:gimhaneul24@gmail.com)
