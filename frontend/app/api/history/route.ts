// frontend/app/api/history/route.ts
import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { db } from "@/lib/db";
import type { FullAnalyzeResult } from "@/app/api/analyze-full/route";

export interface HistoryListItem {
  id: number;
  title: string;
  created_at: string;
  total_clauses: number;
  high_count: number;
  medium_count: number;
  low_count: number;
}

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const { rows } = await db.query(
    `SELECT
       id,
       title,
       created_at,
       (result->>'total_clauses')::int AS total_clauses,
       (result->>'high_count')::int AS high_count,
       (result->>'medium_count')::int AS medium_count,
       (result->>'low_count')::int AS low_count
     FROM analyses
     WHERE user_id = $1
     ORDER BY created_at DESC`,
    [session.user.id]
  );

  return NextResponse.json({ items: rows as HistoryListItem[] });
}

export async function POST(req: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const body = await req.json();
  const title = typeof body.title === "string" && body.title.trim() ? body.title.trim() : "제목 없는 분석";
  const result = body.result as FullAnalyzeResult | undefined;

  if (!result || typeof result.total_clauses !== "number" || !Array.isArray(result.clauses)) {
    return NextResponse.json({ error: "저장할 분석 결과가 올바르지 않습니다." }, { status: 400 });
  }

  const { rows } = await db.query(
    `INSERT INTO analyses (user_id, title, result) VALUES ($1, $2, $3) RETURNING id, created_at`,
    [session.user.id, title, JSON.stringify(result)]
  );

  return NextResponse.json({ id: rows[0].id, created_at: rows[0].created_at }, { status: 201 });
}
