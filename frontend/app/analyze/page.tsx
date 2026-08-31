// frontend/app/analyze/page.tsx
import dynamic from "next/dynamic";
import type { Metadata } from "next";
import { AnalyzeShell } from "./AnalyzeShell";

export const metadata: Metadata = {
  title: "계약서 분석 — Verilex",
  description: "계약서 전문을 업로드하거나 붙여넣어 조항별 법적 리스크를 AI로 분석합니다.",
};

const ContractAnalyzer = dynamic(
  () => import("./ContractAnalyzer").then((m) => m.ContractAnalyzer),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 rounded-full border-2 border-navy border-t-transparent animate-spin" />
      </div>
    ),
  }
);

export default function AnalyzePage() {
  return (
    <AnalyzeShell
      title="계약서 전체 분석"
      description="계약서 전문을 업로드하거나 붙여넣으면 AI가 조항을 자동으로 분리하고 각 조항의 법적 리스크를 분석합니다."
    >
      <ContractAnalyzer />
    </AnalyzeShell>
  );
}
