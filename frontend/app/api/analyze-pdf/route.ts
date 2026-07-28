// frontend/app/api/analyze-pdf/route.ts
import { NextRequest, NextResponse } from "next/server";
import { FASTAPI_URL } from "@/lib/config";

export async function POST(req: NextRequest) {
  const form = await req.formData();

  try {
    const upstream = await fetch(`${FASTAPI_URL}/api/analyze-pdf`, {
      method: "POST",
      headers: process.env.BACKEND_API_KEY ? { "X-API-Key": process.env.BACKEND_API_KEY } : undefined,
      body: form,
    });

    if (!upstream.ok) {
      const err = await upstream.json().catch(() => ({}));
      return NextResponse.json(
        { error: (err as { detail?: string }).detail ?? "분석 서버 오류" },
        { status: upstream.status }
      );
    }

    const data = await upstream.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "분석 서버에 연결할 수 없습니다." },
      { status: 503 }
    );
  }
}
