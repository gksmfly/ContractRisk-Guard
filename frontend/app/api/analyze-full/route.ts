// frontend/app/api/analyze-full/route.ts
import { NextRequest, NextResponse } from "next/server";
import type { RiskLevel, Domain } from "@/types";
import { FASTAPI_URL } from "@/lib/config";

interface ClauseResult {
  id: number;
  original: string;
  domain: Domain;
  risk_level: RiskLevel;
  // 백엔드가 원시 확률(confidence: number) 대신 구간을 보낸다 — KoELECTRA softmax는
  // 보정이 안 돼 있어(ECE 0.289) %로 표시하면 실제보다 30%p 이상 과신하게 된다.
  // confidence_band_accuracy는 그 구간의 실측 정확도이므로, "높음"만 단독으로
  // 보여주지 말고 이 값을 함께 표시할 것.
  // 근거: backend/eval/confidence_calibration.py
  confidence_band: "높음" | "중간" | "낮음";
  confidence_band_accuracy: number;
  evidence_spans: { text: string; start: number; end: number }[];
  legal_basis: { law: string; article: string; description: string }[];
  reasoning: string;
  verified: boolean;
  redteam_note: string;
  evidence_verified: boolean;
}

export interface FullAnalyzeResult {
  total_clauses: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  clauses: ClauseResult[];
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
