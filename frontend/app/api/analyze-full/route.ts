// frontend/app/api/analyze-full/route.ts
import { NextRequest, NextResponse } from "next/server";
import type { RiskLevel, Domain } from "@/types";
import { FASTAPI_URL } from "@/lib/config";

interface ClauseResult {
  id: number;
  original: string;
  domain: Domain;
  risk_level: RiskLevel;
  confidence: number;
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
