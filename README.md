# ContractRisk-Guard

AI-powered Korean contract risk analyzer using KoELECTRA fine-tuning and Forward-Backward consistency verification.

## Overview

ContractRisk-Guard is an automated system that detects unfair or risky clauses in Korean contracts. It combines domain-specific legal knowledge filtering with a fine-tuned KoELECTRA classification model, validated through a Forward-Backward consistency check to minimize hallucination.

## Pipeline

1. **Domain Filtering** — Filters relevant laws, precedents, and interpretations from legal databases
2. **Preprocessing** — Cleans and segments contract text
3. **Seed Labeling** — Generates initial labels using LLM-based grounding
4. **FB-Check** — Forward-Backward consistency verification to filter noisy labels
5. **Model Training** — Fine-tunes KoELECTRA on verified labels
6. **API** — FastAPI server exposing contract analysis endpoints

## Tech Stack

- **Backend**: Python, FastAPI, KoELECTRA (transformers)
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **Infrastructure**: Docker

## Project Structure

```
backend/
├── api/          # FastAPI server & routers
├── domain/       # Legal domain filtering
├── preprocess/   # Text preprocessing
├── labeling/     # Seed labeling
├── fb_check/     # Forward-Backward consistency check
├── model/        # KoELECTRA model wrapper
└── training/     # Fine-tuning pipeline
frontend/         # Next.js UI
scripts/          # Data crawling scripts
docker/           # Docker Compose
```
