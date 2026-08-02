// frontend/app/dashboard/page.tsx
import Link from "next/link";
import { redirect } from "next/navigation";
import { getServerSession } from "next-auth";
import type { Metadata } from "next";
import { LayoutGrid, FileText, Zap, BarChart3, Settings } from "lucide-react";
import { Logo } from "@/components/Logo";
import { CountUp } from "@/components/CountUp";
import { authOptions } from "@/lib/auth";

export const metadata: Metadata = {
  title: "분석 대시보드 — Verilex",
};

const SIDEBAR = [
  { icon: LayoutGrid, label: "대시보드", href: "/dashboard", active: true },
  { icon: FileText, label: "분석 내역", href: "#" },
  { icon: Zap, label: "새 분석", href: "/analyze" },
  { icon: BarChart3, label: "리포트", href: "#" },
  { icon: Settings, label: "설정", href: "#" },
];

const STATS = [
  { value: "24", label: "총 분석 건수", note: "이번 달 +7", color: "text-navy" },
  { value: "3", label: "고위험 조항 발견", note: "즉시 검토 필요", color: "text-seal" },
  { value: "12", label: "수정 완료된 조항", note: "이번 달 처리", color: "text-forest" },
  { value: "98%", label: "법령 근거 검증률", note: "Evidence Verification", color: "text-ochre" },
];

// 데모용 mock 데이터 — 실제 분석 이력 저장·조회 API는 아직 없습니다.
const FILES = [
  { name: "쿠팡_로켓와우_이용약관_2024.pdf", high: 3, medium: 7, low: 12, date: "2024.01.15", report: true },
  { name: "배달의민족_서비스이용약관_v3.2.pdf", high: 1, medium: 4, low: 18, date: "2024.01.12", report: false },
  { name: "토스_금융서비스_이용약관.pdf", high: 0, medium: 6, low: 9, date: "2024.01.10", report: false },
  { name: "카카오페이_결제서비스_약관.pdf", high: 2, medium: 3, low: 14, date: "2024.01.08", report: false },
];

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

      <div className="flex">
        {/* Sidebar */}
        <nav aria-label="대시보드 메뉴" className="hidden md:block w-56 shrink-0 border-r border-rule px-3 py-6 space-y-1">
          {SIDEBAR.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                item.active ? "bg-navy-soft text-navy font-medium" : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              <item.icon className="h-4 w-4" aria-hidden />
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Main */}
        <main className="flex-1 px-6 py-8 md:px-10">
          <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
            <h1 className="text-2xl font-bold text-slate-900">분석 대시보드</h1>
            <Link
              href="/analyze"
              className="text-xs font-semibold bg-navy hover:opacity-90 text-white rounded-lg px-4 py-2.5 transition-opacity"
            >
              + 새 약관 분석
            </Link>
          </div>

          {/* Stat cards */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {STATS.map((s) => (
              <div key={s.label} className="border border-rule p-5">
                <p className={`text-3xl font-bold ${s.color}`}>
                  <CountUp value={s.value} />
                </p>
                <p className="text-sm text-slate-700 mt-1">{s.label}</p>
                <p className="text-xs text-slate-400 mt-0.5">{s.note}</p>
              </div>
            ))}
          </div>

          {/* Recent files table */}
          <div className="border border-rule">
            <div className="flex items-center justify-between px-5 py-3 border-b border-rule">
              <p className="text-sm font-semibold text-slate-900">최근 분석 파일</p>
              <Link href="#" className="text-xs text-navy hover:underline">전체 보기 →</Link>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-slate-400 border-b border-rule">
                    <th className="font-normal py-2.5 px-5">파일명</th>
                    <th className="font-normal py-2.5 px-2 text-center">고위험</th>
                    <th className="font-normal py-2.5 px-2 text-center">중위험</th>
                    <th className="font-normal py-2.5 px-2 text-center">저위험</th>
                    <th className="font-normal py-2.5 px-2">상태</th>
                    <th className="font-normal py-2.5 px-2">분석일</th>
                    <th className="font-normal py-2.5 px-5" />
                  </tr>
                </thead>
                <tbody>
                  {FILES.map((f) => (
                    <tr key={f.name} className="border-b border-rule last:border-0 hover:bg-slate-50">
                      <td className="py-3 px-5 flex items-center gap-2 font-medium text-slate-900">
                        <FileText className="h-3.5 w-3.5 text-slate-400 shrink-0" aria-hidden />
                        {f.name}
                      </td>
                      <td className="text-center px-2">
                        <span className="text-seal font-semibold">{f.high}</span>
                      </td>
                      <td className="text-center px-2">
                        <span className="text-ochre font-semibold">{f.medium}</span>
                      </td>
                      <td className="text-center px-2">
                        <span className="text-forest font-semibold">{f.low}</span>
                      </td>
                      <td className="px-2">
                        <span className="bg-forest-soft text-forest text-xs font-medium rounded-full px-2 py-0.5">
                          완료
                        </span>
                      </td>
                      <td className="px-2 text-slate-500 text-xs">{f.date}</td>
                      <td className="px-5 text-right whitespace-nowrap">
                        {f.report && (
                          <Link
                            href="/report-preview"
                            className="text-xs text-navy hover:underline mr-3"
                          >
                            PDF 리포트
                          </Link>
                        )}
                        <Link
                          href="/analyze"
                          className="text-xs border border-rule hover:border-navy hover:text-navy rounded-lg px-3 py-1.5 transition-colors"
                        >
                          열기
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <p className="text-xs text-slate-400 mt-4">
            데모 화면입니다 — 실제 분석 이력 저장·조회 기능은 아직 백엔드에 연동되지 않았습니다.
            {!session && " (지금은 Google 로그인 없이 보는 개발용 미리보기입니다 — 배포 환경에서는 로그인이 필요합니다.)"}
          </p>
        </main>
      </div>
    </div>
  );
}
