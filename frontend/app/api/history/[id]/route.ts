// frontend/app/api/history/[id]/route.ts
import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { db } from "@/lib/db";

export async function GET(req: NextRequest, { params }: { params: { id: string } }) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const id = Number(params.id);
  if (!Number.isInteger(id)) {
    return NextResponse.json({ error: "잘못된 요청입니다." }, { status: 400 });
  }

  // user_id까지 WHERE 조건에 넣어서, 다른 사용자의 id를 추측해 넣어도 남의 결과를 못 본다.
  const { rows } = await db.query(
    `SELECT id, title, result, created_at FROM analyses WHERE id = $1 AND user_id = $2`,
    [id, session.user.id]
  );

  if (rows.length === 0) {
    return NextResponse.json({ error: "저장된 분석을 찾을 수 없습니다." }, { status: 404 });
  }

  return NextResponse.json(rows[0]);
}
