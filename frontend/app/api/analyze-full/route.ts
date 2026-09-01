// frontend/app/api/analyze-full/route.ts
import { NextRequest, NextResponse } from "next/server";
import type { RiskLevel, Domain } from "@/types";
import { FASTAPI_URL } from "@/lib/config";

interface ClauseResult {
  id: number;
  original: string;
  // 모델이 지목한 약관규제법 조. **참고값이다** — 조 단위 정밀도(44%대)가
  // 조항 단위 재현(78.0%)보다 훨씬 낮으므로 "제N조 위반입니다"로 단정하지 말고
  // "제N조 관련으로 보입니다"로 표시할 것. 근거: backend/eval/article_gold_eval.py
  articles: string[];
  needs_review: boolean;
  // 아래 셋은 **옛 응답 호환용**이다. 2026-08-31부터 백엔드가 위험도 3단계와
  // 신뢰도 구간을 내지 않는다(조 multi-label 모델에 risk 헤드가 없고, 구간 정확도는
  // models/v4 전용 실측값이라 옮길 수 없다). 새 응답에서는 비어 있다.
  domain?: Domain;
  risk_level?: RiskLevel;
  confidence_band?: "높음" | "중간" | "낮음";
  confidence_band_accuracy?: number;
  evidence_spans: { text: string; start: number; end: number }[];
  legal_basis: { law: string; article: string; description: string }[];
  reasoning: string;
  verified: boolean;
  redteam_note: string;
  evidence_verified: boolean;
}

export interface FullAnalyzeResult {
  total_clauses: number;
  // 확인이 필요하다고 판단된 조항 수. 위험도 3단계를 내지 않으므로 세 칸이 아니라 하나다.
  review_count: number;
  // 옛 저장분 호환용 — 새 응답에서는 항상 0이다. 화면에서 읽지 말 것.
  high_count?: number;
  medium_count?: number;
  low_count?: number;
  clauses: ClauseResult[];
  // 입력에서 분리된 조항 수. total_clauses(=확인 필요 판정 수)와 다르다 —
  // 나머지는 out_of_scope로 빠지며 "안전"이 아니라 "확인되지 않음"이다.
  input_clauses?: number;
  truncated_clauses?: number;
  out_of_scope?: { id: number; original: string; reason: string }[];
  // 판단을 낸 체크포인트 이름. analyses.result(JSONB)에 통째로 저장되므로
  // 나중에 `WHERE result->>'model_version' = 'v4'`로 옛 모델 결과를 골라낼 수 있다.
  // 예전 저장분에는 없으므로 optional.
  model_version?: string;
}

export async function POST(req: NextRequest) {
  const body = await req.json();

  if (!body.text || body.text.trim().length < 20) {
    return NextResponse.json(
      { error: "계약서 내용이 너무 짧습니다." },
      { status: 400 }
    );
  }

  try {
    const upstream = await fetch(`${FASTAPI_URL}/api/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.BACKEND_API_KEY ? { "X-API-Key": process.env.BACKEND_API_KEY } : {}),
      },
      body: JSON.stringify({ text: body.text }),
    });

    if (!upstream.ok) {
      const err = await upstream.json().catch(() => ({}));
      return NextResponse.json(
        { error: (err as { detail?: string }).detail ?? "분석 서버 오류" },
        { status: upstream.status }
      );
    }

    const data: FullAnalyzeResult = await upstream.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "분석 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요." },
      { status: 503 }
    );
  }
}
