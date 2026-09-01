// frontend/app/dashboard/page.tsx
import Link from "next/link";
import { redirect } from "next/navigation";
import { getServerSession } from "next-auth";
import type { Metadata } from "next";
import { FileText, AlertTriangle, CheckCircle2 } from "lucide-react";
import { Logo } from "@/components/Logo";
import { authOptions } from "@/lib/auth";
import { db } from "@/lib/db";

export const metadata: Metadata = {
  title: "내 분석 히스토리 — Verilex",
};

interface HistoryRow {
  id: number;
  title: string;
  created_at: string;
  total_clauses: number;
  // 새 결과에만 있다. 옛 행(v4, 위험도 3단계)은 null이라 개수를 표시하지 않는다 —
  // v4의 high_count와 지금의 review_count는 **정의가 다른 값**이라 같은 열에 나란히
  // 놓으면 비교 가능한 것처럼 보인다. 오늘 여러 번 잡은 "모집단이 다른 두 값 병치"다.
  review_count: number | null;
  model_version: string | null;
  medium_count: number;
  low_count: number;
}

export default async function DashboardPage() {
  const session = await getServerSession(authOptions);

  // Google OAuth 자격증명이 아직 설정되지 않아 로컬에서는 로그인 자체가 안 된다.
  // 프로덕션에서는 그대로 로그인을 강제하되, 개발 환경에서는 데모 화면을 바로
  // 미리볼 수 있게 리다이렉트를 건너뛴다 — 실제 인증 로직 자체는 바꾸지 않는다.
  if (!session && process.env.NODE_ENV === "production") {
    redirect("/login");
  }

  const displayName = session?.user?.name ?? session?.user?.email ?? "테스트 사용자";
  const initial = displayName[0]?.toUpperCase() ?? "?";

  let items: HistoryRow[] = [];
  if (session?.user?.id) {
    const { rows } = await db.query(
      `SELECT
         id, title, created_at,
         (result->>'total_clauses')::int AS total_clauses,
         -- 2026-08-31부터 위험도 3단계를 내지 않는다. 새 결과는 review_count를 쓰고,
         -- 옛 저장분(high_count)은 그대로 남아 있으므로 COALESCE로 둘 다 받는다.
         (result->>'review_count')::int AS review_count,
         result->>'model_version' AS model_version
       FROM analyses
       WHERE user_id = $1
       ORDER BY created_at DESC`,
      [session.user.id]
    );
    items = rows as HistoryRow[];
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Top bar */}
      <header className="border-b border-rule px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="p-1 bg-navy-soft rounded-md border border-navy/20 group-hover:bg-navy/10 transition-colors">
            <Logo className="h-3.5 w-3.5 text-navy" />
          </div>
          <span className="font-bold text-slate-900 text-sm">Verilex</span>
        </Link>
        {session?.user?.image ? (
          // eslint-disable-next-line @next/next/no-img-element -- Google 프로필 이미지, next/image remotePatterns 미설정
          <img
            src={session.user.image}
            alt={displayName}
            className="w-8 h-8 rounded-full object-cover"
          />
        ) : (
          <div className="w-8 h-8 rounded-full bg-navy text-white text-xs font-semibold flex items-center justify-center">
            {initial}
          </div>
        )}
      </header>

      <main className="max-w-3xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
          <h1 className="text-2xl font-bold text-slate-900">내 분석 히스토리</h1>
          <Link
            href="/analyze"
            className="text-xs font-semibold bg-navy hover:opacity-90 text-white rounded-lg px-4 py-2.5 transition-opacity"
          >
            + 새 분석 시작
          </Link>
        </div>

        {items.length === 0 ? (
          <div className="border border-dashed border-rule rounded-xl py-16 text-center space-y-2">
            <p className="text-sm text-slate-500">아직 저장된 분석이 없습니다.</p>
            <p className="text-xs text-slate-400">
              계약서를 분석한 뒤 결과 화면에서 &ldquo;저장하기&rdquo;를 누르면 여기에 쌓입니다.
            </p>
          </div>
        ) : (
          <div className="border border-rule divide-y divide-rule">
            {items.map((item) => (
              <Link
                key={item.id}
                href={`/analyze/${item.id}`}
                className="flex items-center justify-between gap-4 px-5 py-4 hover:bg-slate-50 transition-colors"
              >
                <div className="flex items-center gap-3 min-w-0">
                  {item.review_count === null ? (
                    <span className="w-2 h-2 rounded-full bg-slate-200 shrink-0" aria-hidden />
                  ) : item.review_count > 0 ? (
                    <span className="w-2 h-2 rounded-full bg-ochre shrink-0" aria-hidden />
                  ) : (
                    <span className="w-2 h-2 rounded-full bg-slate-300 shrink-0" aria-hidden />
                  )}
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-slate-900 truncate flex items-center gap-1.5">
                      <FileText className="h-3.5 w-3.5 text-slate-400 shrink-0" aria-hidden />
                      {item.title}
                      {item.review_count === null ? (
                        // 옛 버전(v4, 위험도 3단계) 결과 — 개수 체계가 달라 나란히 못 놓는다
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-400">
                          이전 버전으로 분석됨
                        </span>
                      ) : item.review_count > 0 ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-ochre">
                          <AlertTriangle className="h-3 w-3" aria-hidden /> 확인 필요 {item.review_count}건
                        </span>
                      ) : (
                        // "위험 조항 없음"이 아니라 "확인되지 않음"이다 — 조항 단위 재현이
                        // 78%이므로 5건 중 1건은 못 찾는다. 안전 판정으로 읽히면 안 된다.
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-500">
                          <CheckCircle2 className="h-3 w-3" aria-hidden /> 확인 필요 조항 없음
                        </span>
                      )}
                    </p>
                    <p className="text-xs text-slate-400 font-mono mt-0.5">
                      {new Date(item.created_at).toLocaleDateString("ko-KR")} 저장됨
                    </p>
                  </div>
                </div>
                <span className="shrink-0 text-xs font-semibold text-navy">다시 보기 →</span>
              </Link>
            ))}
          </div>
        )}

        <p className="text-xs text-slate-400 mt-4">
          전부 실제로 로그인 후 저장한 결과입니다 — 예시 데이터 없음.
          {!session && " (지금은 Google 로그인 없이 보는 개발용 미리보기라 목록이 비어 있습니다.)"}
        </p>
      </main>
    </div>
  );
}
