# ContractRisk-Guard: Contract Clause Risk Prediction via Forward-Backward Consistency Check

> **TL;DR**: Detects high-risk termination and liability-limitation clauses in Korean service contracts using a 3-stage FB-Check pipeline to filter LLM hallucinations from auto-labeled training data, then deploys a fine-tuned KoELECTRA model as the core inference engine within a 6-agent multi-agent system — published at KAICTS 2025.

---

## Problem Statement

Korean service contracts frequently contain unfair termination clauses (약관규제법 §9) and liability limitation clauses (약관규제법 §7) that cause legal disputes, yet are too dense for non-experts to review manually.

- Existing LLM-based legal systems lack a verification layer for generated judgments
- LLM auto-labeling creates self-consistency bias: errors accumulate when the same model family validates its own outputs
- Legal-domain mislabeled data compounds through retraining, degrading both reliability and explainability

---

## Approach

- **FB-Check to filter hallucinations from auto-labeled data**: Forward Labeling (GPT-4o generates label L and evidence span E from clause C) → Backward Grounding (1st-generation KoELECTRA validates E ⊂ C at index level) → Consistency Verification (re-labels from E alone; L′ == L → CLEAN, otherwise NOISE)
- **Data Flywheel**: CLEAN data trains a 2nd-generation model that replaces the Backward Grounding agent, iteratively co-optimizing data quality and model generalization
- **KoELECTRA as the inference engine**: The fine-tuned classifier makes all risk judgments; GPT-4o is strictly limited to explanation generation and search query construction — reducing LLM calls, latency, and cost while keeping judgments consistent
- **6-agent multi-agent collaboration**: Analysis Agent (clause type + initial risk + key terms) → Retrieval Strategy Agent (Dense + Sparse Hybrid Retrieval, metadata filtering) → Evidence Selection Agent (reranking candidate docs) → Judgment Agent (fine-tuned KoELECTRA) → Red-team Agent (adversarial case search to probe judgment bias) → Evidence Verification Agent (semantic coherence check + re-search control if evidence is insufficient)
- **FTC enforcement cases as seed data**: 공정거래위원회 시정조치 cases are objectively High-risk (public enforcement = confirmed violation), avoiding annotator subjectivity

---

## Key Results

| Metric | Value |
|--------|-------|
| CLEAN label match rate | 94.9% (131 / 138 cases) |
| NOISE label match rate | 76.2% (439 / 576 cases) |
| Risk categories | High / Medium / Low |
| Clause domains | 해지 조항, 책임제한 조항 |
| Legal corpus | 16 laws, 26 interpretations, 1,995 court precedents, 2,488+ FTC cases |
| Published | KAICTS 2025 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python, uvicorn |
| Core Inference | KoELECTRA fine-tuned on CLEAN data (dual-head: domain + risk) |
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
│   ├── labeling/seed.py            # FTC cases + standard contracts → seed data
│   ├── training/train.py           # KoELECTRA fine-tuning on CLEAN data
│   ├── model/electra.py            # DualHeadElectra (domain + risk heads)
│   └── scripts/                    # Law API crawler, FTC case scraper
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # Landing page
│   │   └── analyze/                # Contract analyzer UI
│   └── components/
└── docker/docker-compose.yml       # PostgreSQL pgvector
```

---

## Getting Started

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env
# Set: OPENAI_API_KEY, DATABASE_URL

uvicorn api.server:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## Limitations & Future Work

- Quantitative performance evaluation on real standard contract datasets is pending
- Coverage gaps remain for clauses involving recent law amendments or sparse precedent domains
- Future: visualization structures for explainability; adaptive re-search depth control in the Evidence Verification Agent; extension to additional clause types

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Seoyeon Kim** | Undergraduate Researcher, Korean Bible University  
[GitHub](https://github.com/gksmfly) · [Email](mailto:gimhaneul24@gmail.com)
