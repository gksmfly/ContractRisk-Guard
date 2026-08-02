// frontend/app/report-preview/page.tsx
"use client";

import Link from "next/link";
import { ChevronLeft, Printer, Download } from "lucide-react";

// 데모용 mock 데이터 — 대시보드의 "쿠팡_로켓와우_이용약관_2024.pdf" 예시와 동일하다.
const REPORT = {
  fileName: "쿠팡_로켓와우_이용약관_2024.pdf",
  analyzedAt: "2024년 1월 15일",
  engine: "Verilex v2.1",
  totalClauses: 22,
  high: 3,
  medium: 7,
  low: 12,
  summary:
    "분석 대상 약관에서 총 22개 조항이 검토되었으며, 3개 조항이 약관규제법 위반 가능성이 높은 고위험으로 분류되었습니다. 특히 제12조(서비스 이용 제한), 제15조(손해배상 책임 제한), 제18조(일방적 약관 변경)는 즉각적인 수정 검토를 권고합니다.",
};

function handleDownload() {
  const lines = [
    "약관 리스크 분석 감사 리포트",
    `대상 파일: ${REPORT.fileName}`,
    `분석일: ${REPORT.analyzedAt}`,
    `분석 엔진: ${REPORT.engine}`,
    "",
    `종합 판단 요약 — 고위험 ${REPORT.high}건 · 중위험 ${REPORT.medium}건 · 저위험 ${REPORT.low}건`,
    "",
    REPORT.summary,
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "약관-리스크-분석-감사-리포트.txt";
  a.click();
  URL.revokeObjectURL(url);
}

export default function ReportPreviewPage() {
  return (
    <div className="min-h-screen bg-slate-800">
      {/* Toolbar */}
      <header className="bg-navy text-white h-14 flex items-center justify-between px-4 print:hidden">
        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="flex items-center gap-1 text-slate-300 hover:text-white text-sm">
            <ChevronLeft className="h-4 w-4" />홈
          </Link>
          <span className="text-xs text-slate-400 font-mono">1 / 1 페이지 · 100%</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => window.print()}
            className="flex items-center gap-1.5 text-xs border border-white/20 hover:bg-white/10 rounded-lg px-3 py-1.5 transition-colors"
          >
            <Printer className="h-3.5 w-3.5" aria-hidden />
            인쇄
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 text-xs bg-navy hover:opacity-90 rounded-lg px-3 py-1.5 transition-opacity"
          >
            <Download className="h-3.5 w-3.5" aria-hidden />
            다운로드
          </button>
        </div>
      </header>

      {/* Page preview */}
      <main className="py-10 px-4 flex justify-center print:p-0">
        <div className="bg-white w-full max-w-2xl p-10 shadow-xl print:shadow-none">
          <div className="border-t-4 border-navy pt-4">
            <h1 className="text-2xl font-bold text-slate-900">약관 리스크 분석 감사 리포트</h1>
            <p className="text-xs text-slate-500 mt-2">
              대상 파일: <span className="font-medium text-slate-700">{REPORT.fileName}</span>
              {" · "}분석일: <span className="font-medium text-slate-700">{REPORT.analyzedAt}</span>
              {" · "}분석 엔진: <span className="font-medium text-slate-700">{REPORT.engine}</span>
            </p>
          </div>

          <div className="mt-8 pt-6 border-t border-rule">
            <h2 className="text-sm font-bold text-navy mb-3">종합 판단 요약</h2>
            <div className="grid grid-cols-3 gap-3 mb-5">
              <div className="bg-seal-soft p-4 text-center">
                <p className="text-2xl font-bold text-seal">{REPORT.high}건</p>
                <p className="text-xs text-slate-600 mt-0.5">고위험</p>
              </div>
              <div className="bg-ochre-soft p-4 text-center">
                <p className="text-2xl font-bold text-ochre">{REPORT.medium}건</p>
                <p className="text-xs text-slate-600 mt-0.5">중위험</p>
              </div>
              <div className="bg-forest-soft p-4 text-center">
                <p className="text-2xl font-bold text-forest">{REPORT.low}건</p>
                <p className="text-xs text-slate-600 mt-0.5">저위험</p>
              </div>
            </div>
            <p className="text-sm text-slate-700 leading-relaxed">{REPORT.summary}</p>
          </div>

          <div className="mt-10 pt-4 border-t border-rule flex justify-between text-[11px] text-slate-400">
            <span>Verilex — 법령 근거 기반 약관 리스크 분석</span>
            <span>1 / 1</span>
          </div>
        </div>
      </main>

      <p className="text-center text-xs text-slate-400 pb-8 print:hidden">
        데모 미리보기입니다 — 표시된 수치는 예시이며 실제 분석 결과와 연동되어 있지 않습니다.
        실제 조항별 리포트는 <Link href="/analyze" className="underline hover:text-white">계약서 분석</Link>에서 PDF 리포트 버튼으로 생성하세요.
      </p>
    </div>
  );
}
