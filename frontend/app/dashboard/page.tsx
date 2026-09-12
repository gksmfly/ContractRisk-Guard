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
  // 놓으면 비교 가능한 것처럼 보인다.
  review_count: number | null;
  model_version: string | null;
}

interface Totals {
  analyses: number;
  clauses: number;
  flagged: number;
}

// 화면에 고정된 순서로 세운다 — 데이터에 없는 조도 0으로 자리를 지켜야 분석을 거듭해도
// 막대의 위치가 안 흔들린다. 제13조는 학습 표본 부족으로 헤드에서 접혀 예측되지 않으므로
// 축에서도 뺀다(0으로 두면 "한 번도 안 걸린 조"로 오독된다).
const ARTICLES: { id: string; title: string }[] = [
  { id: "제6조", title: "일반원칙" },
  { id: "제7조", title: "면책조항의 금지" },
  { id: "제8조", title: "손해배상액의 예정" },
  { id: "제9조", title: "계약의 해제·해지" },
  { id: "제10조", title: "채무의 이행" },
  { id: "제11조", title: "고객의 권익 보호" },
  { id: "제12조", title: "의사표시의 의제" },
  { id: "제14조", title: "소송 제기의 금지 등" },
];

export default async function DashboardPage() {
  const session = await getServerSession(authOptions);

  // 개발 환경에서는 로그인 없이 화면을 미리볼 수 있게 리다이렉트를 건너뛴다.
  // 인증 로직 자체는 그대로다 — 프로덕션에서는 반드시 로그인을 강제한다.
  if (!session && process.env.NODE_ENV === "production") {
    redirect("/login");
  }

  const displayName = session?.user?.name ?? session?.user?.email ?? "테스트 사용자";
  const initial = displayName[0]?.toUpperCase() ?? "?";

  let items: HistoryRow[] = [];
  let totals: Totals = { analyses: 0, clauses: 0, flagged: 0 };
  let byArticle = new Map<string, number>();

  if (session?.user?.id) {
    const uid = session.user.id;

    const history = await db.query(
      `SELECT
         id, title, created_at,
         (result->>'total_clauses')::int AS total_clauses,
         -- 2026-08-31부터 위험도 3단계를 내지 않는다. 새 결과는 review_count를 쓴다.
         (result->>'review_count')::int AS review_count,
         -- get_model_version 은 필드명이 잘못 나가던 기간(2026-09-01 이전)의 저장분이다.
         -- 그 창에 저장된 행도 세대를 알 수 있게 둘 다 받는다.
         COALESCE(result->>'model_version', result->>'get_model_version') AS model_version
       FROM analyses
       WHERE user_id = $1
       ORDER BY created_at DESC`,
      [uid]
    );
    items = history.rows as HistoryRow[];

    const agg = await db.query(
      `SELECT
         count(*)::int                                            AS analyses,
         COALESCE(sum((result->>'total_clauses')::int), 0)::int   AS clauses,
         COALESCE(sum((result->>'review_count')::int), 0)::int    AS flagged
       FROM analyses WHERE user_id = $1`,
      [uid]
    );
    totals = agg.rows[0] as Totals;

    // 조별 지목 횟수. 옛 저장분에는 clauses[].articles 자체가 없으므로 jsonb_typeof로
    // 막아둔다 — 막지 않으면 v4 행 하나에 쿼리 전체가 에러로 죽는다.
    const dist = await db.query(
      `SELECT art AS article, count(*)::int AS n
         FROM analyses an
         CROSS JOIN LATERAL jsonb_array_elements(
           CASE WHEN jsonb_typeof(an.result->'clauses') = 'array'
                THEN an.result->'clauses' ELSE '[]'::jsonb END) AS c
         CROSS JOIN LATERAL jsonb_array_elements_text(
           CASE WHEN jsonb_typeof(c->'articles') = 'array'
                THEN c->'articles' ELSE '[]'::jsonb END) AS art
        WHERE an.user_id = $1
        GROUP BY art`,
      [uid]
    );
    byArticle = new Map(
      (dist.rows as { article: string; n: number }[]).map((r) => [r.article, r.n])
    );
  }

  const counts = ARTICLES.map((a) => ({ ...a, n: byArticle.get(a.id) ?? 0 }));
  const peak = Math.max(1, ...counts.map((c) => c.n));
  const hasDist = counts.some((c) => c.n > 0);
  const flagRate = totals.clauses > 0 ? (totals.flagged / totals.clauses) * 100 : null;

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
        <div className="flex items-center justify-between mb-7 flex-wrap gap-3">
          <h1 className="text-2xl font-bold text-slate-900">내 분석 히스토리</h1>
          <Link
            href="/analyze"
            className="text-xs font-semibold bg-navy hover:opacity-90 text-white rounded-lg px-4 py-2.5 transition-opacity"
          >
            + 새 분석 시작
          </Link>
        </div>

        {/* ── 요약 ─────────────────────────────────────────────────────────
            숫자 셋만 둔다. 분석 건수가 한 자릿수인 동안 추이 그래프는 선이 아니라
            점 두 개라, 없는 경향을 그려 보이게 된다. */}
        <section aria-label="요약" className="grid grid-cols-3 border border-rule divide-x divide-rule">
          {[
            { label: "분석한 계약서", value: totals.analyses, unit: "건", tone: "ink" },
            { label: "검토한 조항", value: totals.clauses, unit: "개", tone: "ink" },
            { label: "확인 필요 조항", value: totals.flagged, unit: "개", tone: "flag" },
          ].map((s) => (
            <div key={s.label} className="px-5 py-4">
              <p className="text-[11px] font-medium text-slate-500 tracking-wide">{s.label}</p>
              <p
                className={`mt-1.5 font-bold tabular-nums leading-none ${
                  s.tone === "flag" ? "text-ochre" : "text-slate-900"
                }`}
              >
                <span className="text-3xl">{s.value.toLocaleString("ko-KR")}</span>
                <span className="text-sm font-semibold ml-1 opacity-60">{s.unit}</span>
              </p>
            </div>
          ))}
        </section>

        {flagRate !== null && (
          <p className="text-xs text-slate-500 mt-2">
            검토한 조항의{" "}
            <span className="font-semibold text-ochre tabular-nums">{flagRate.toFixed(1)}%</span>가
            확인 필요로 표시됐습니다. 조항 단위 재현율이 78%라 <b className="font-semibold">놓친 조항이 있을 수 있습니다</b>.
          </p>
        )}

        {/* ── 조별 분포 ────────────────────────────────────────────────────
            단일 계열이라 막대 색을 나누지 않는다 — 크기는 길이로 이미 인코딩돼 있고,
            여기에 색까지 얹으면 같은 값을 두 번 말하는 셈이다. */}
        <section aria-label="조별 지목 분포" className="mt-8">
          <div className="flex items-baseline justify-between gap-3 mb-3">
            <h2 className="text-sm font-bold text-slate-900">약관규제법 조별 지목 횟수</h2>
            <span className="text-[11px] text-slate-400">누적 {totals.flagged}개 조항</span>
          </div>

          {!hasDist ? (
            <div className="border border-dashed border-rule rounded-xl py-10 text-center">
              <p className="text-sm text-slate-500">아직 지목된 조항이 없습니다.</p>
              <p className="text-xs text-slate-400 mt-1">
                계약서를 분석하고 저장하면 어떤 조에 몰리는지 여기에 쌓입니다.
              </p>
            </div>
          ) : (
            <ol className="border border-rule divide-y divide-rule">
              {counts.map((c) => (
                <li key={c.id} className="flex items-center gap-3 px-4 py-2.5">
                  <span className="w-[3.5rem] shrink-0 text-xs font-semibold text-slate-900 tabular-nums">
                    {c.id}
                  </span>
                  <span className="w-[7.5rem] shrink-0 text-[11px] text-slate-500 truncate hidden sm:block">
                    {c.title}
                  </span>
                  <span className="flex-1 h-2.5 bg-slate-100 rounded-sm overflow-hidden">
                    {c.n > 0 && (
                      <span
                        className="block h-full bg-navy rounded-r-sm"
                        style={{ width: `${Math.max(4, (c.n / peak) * 100)}%` }}
                      />
                    )}
                  </span>
                  <span
                    className={`w-7 shrink-0 text-right text-xs tabular-nums ${
                      c.n > 0 ? "font-semibold text-slate-900" : "text-slate-300"
                    }`}
                  >
                    {c.n}
                  </span>
                </li>
              ))}
            </ol>
          )}
          <p className="text-[11px] text-slate-400 mt-2">
            제13조는 학습 표본이 부족해 모델 출력에서 제외돼 있어 축에도 없습니다.
            조 이름은 <b className="font-medium">참고값</b>입니다 — 조 단위 정확도는 아직 검증되지 않았습니다.
          </p>
        </section>

        {/* ── 히스토리 ────────────────────────────────────────────────── */}
        <section aria-label="분석 목록" className="mt-8">
          <h2 className="text-sm font-bold text-slate-900 mb-3">분석 기록</h2>

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
        </section>

        <p className="text-xs text-slate-400 mt-4">
          전부 실제로 로그인 후 저장한 결과입니다 — 예시 데이터 없음.
          {!session && " (지금은 Google 로그인 없이 보는 개발용 미리보기라 목록이 비어 있습니다.)"}
        </p>
      </main>
    </div>
  );
}
