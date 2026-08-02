// frontend/app/api/analyze-stream/route.ts
import { NextRequest } from "next/server";
import { FASTAPI_URL } from "@/lib/config";

// backend의 /api/analyze/stream(SSE)을 그대로 프록시한다. analyze-full/route.ts와
// 달리 응답을 JSON으로 모으지 않고 upstream.body(ReadableStream)를 그대로 흘려보내
// 프론트가 조항이 끝날 때마다 오는 이벤트를 실시간으로 받게 한다.
export async function POST(req: NextRequest) {
  const body = await req.json();

  if (!body.text || body.text.trim().length < 20) {
    return new Response(JSON.stringify({ error: "계약서 내용이 너무 짧습니다." }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${FASTAPI_URL}/api/analyze/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.BACKEND_API_KEY ? { "X-API-Key": process.env.BACKEND_API_KEY } : {}),
      },
      body: JSON.stringify({ text: body.text }),
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
