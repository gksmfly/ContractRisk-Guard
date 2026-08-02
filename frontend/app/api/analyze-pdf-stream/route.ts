// frontend/app/api/analyze-pdf-stream/route.ts
import { NextRequest } from "next/server";
import { FASTAPI_URL } from "@/lib/config";

// backend의 /api/analyze-pdf/stream(SSE)을 그대로 프록시한다. analyze-pdf/route.ts와
// 달리 응답을 JSON으로 모으지 않고 upstream.body를 그대로 흘려보내 PDF도 텍스트
// 경로와 동일하게 조항 단위 실시간 진행률을 받게 한다.
export async function POST(req: NextRequest) {
  const form = await req.formData();

  let upstream: Response;
  try {
    upstream = await fetch(`${FASTAPI_URL}/api/analyze-pdf/stream`, {
      method: "POST",
      headers: process.env.BACKEND_API_KEY ? { "X-API-Key": process.env.BACKEND_API_KEY } : undefined,
      body: form,
    });
  } catch {
    return new Response(
      JSON.stringify({ error: "분석 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요." }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }

  if (!upstream.ok || !upstream.body) {
    const err = await upstream.json().catch(() => ({}));
    return new Response(
      JSON.stringify({ error: (err as { detail?: string }).detail ?? "분석 서버 오류" }),
      { status: upstream.status || 502, headers: { "Content-Type": "application/json" } }
    );
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
