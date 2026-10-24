// frontend/lib/require-session.ts
// 분석 API 라우트(analyze-full / analyze-stream / analyze-pdf / analyze-pdf-stream) 맨 앞에서 부른다.
// 페이지(app/analyze/page.tsx)만 로그인을 확인하면 API 주소로 직접 요청하는 사람은 로그인 없이 분석(GPU·OpenAI)을 쓸 수 있다.
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { isEmailAllowed, loginRequired } from "@/lib/access";

/** 통과하면 null, 막아야 하면 401/403 Response를 돌려준다. */
export async function requireSession(): Promise<Response | null> {
  if (!loginRequired()) return null;
  const session = await getServerSession(authOptions);
  if (!session?.user) return json(401, "로그인이 필요합니다.");
  if (!isEmailAllowed(session.user.email)) return json(403, "접근이 허용되지 않은 계정입니다.");
  return null;
}

function json(status: number, error: string): Response {
  return new Response(JSON.stringify({ error }), { status, headers: { "Content-Type": "application/json" } });
}
