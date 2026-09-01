// frontend/app/analyze/page.tsx
import dynamic from "next/dynamic";
import { redirect } from "next/navigation";
import { getServerSession } from "next-auth";
import type { Metadata } from "next";
import { authOptions } from "@/lib/auth";
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

export default async function AnalyzePage() {
  const session = await getServerSession(authOptions);

  // Google OAuth 자격증명이 아직 없어 로컬에서는 로그인 자체가 안 된다 — /dashboard,
  // /analyze/[id]와 같은 패턴으로 프로덕션에서만 강제하고 개발 환경은 그대로 미리본다.
  if (!session && process.env.NODE_ENV === "production") {
    redirect("/login");
  }

  return (
    <AnalyzeShell
      title="계약서 전체 분석"
      description="계약서 전문을 업로드하거나 붙여넣으면 AI가 조항을 자동으로 분리하고 각 조항의 법적 리스크를 분석합니다."
    >
      <ContractAnalyzer />
    </AnalyzeShell>
  );
}
