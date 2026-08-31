// frontend/app/analyze/AnalyzeShell.tsx
import Link from "next/link";
import { ChevronLeft } from "lucide-react";
import { Logo } from "@/components/Logo";

// /analyze(새 분석)와 /analyze/[id](저장된 결과 다시 보기)가 공유하는 헤더+레이아웃.
export function AnalyzeShell({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-white">
      {/* Dark sticky header */}
      <header className="border-b border-slate-800/60 bg-navy/95 backdrop-blur-md sticky top-0 z-40 print:hidden">
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-1.5 text-slate-400 hover:text-white text-sm transition-colors"
            aria-label="홈으로 돌아가기"
          >
            <ChevronLeft className="h-4 w-4" />
            <span>홈</span>
          </Link>

          <Link href="/" className="flex items-center gap-2 group" aria-label="Verilex 홈으로 이동">
            <div className="p-1 bg-navy/40 rounded-md border border-navy-soft/20 group-hover:bg-navy/60 transition-colors">
              <Logo className="h-3.5 w-3.5 text-navy-soft" />
            </div>
            <span className="font-bold text-white text-sm">
              Veri<span className="text-navy-soft">lex</span>
            </span>
          </Link>

          <div className="w-14" aria-hidden />
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-10 space-y-8">
        {/* Page header */}
        <div className="space-y-2 max-w-2xl mx-auto print:hidden">
          <h1 className="text-2xl font-bold text-slate-900">{title}</h1>
          <p className="text-slate-500 text-sm leading-relaxed">{description}</p>
        </div>

        {children}

        {/* Disclaimer */}
        <p className="text-xs text-slate-400 text-center pb-6 leading-relaxed max-w-2xl mx-auto print:hidden">
          본 서비스의 분석 결과는 법적 조언을 대체하지 않으며 참고용으로만 활용하시기 바랍니다.
          최종 계약서 검토는 반드시 법률 전문가와 함께 진행하세요.
        </p>
      </main>
    </div>
  );
}
