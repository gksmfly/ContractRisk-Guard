// frontend/app/analyze/[id]/page.tsx
import dynamic from "next/dynamic";
import { redirect, notFound } from "next/navigation";
import { getServerSession } from "next-auth";
import type { Metadata } from "next";
import { authOptions } from "@/lib/auth";
import { db } from "@/lib/db";
import { AnalyzeShell } from "../AnalyzeShell";
import type { FullAnalyzeResult } from "@/app/api/analyze-full/route";

export const metadata: Metadata = {
  title: "저장된 분석 결과 — Verilex",
};

const ContractAnalyzer = dynamic(
  () => import("../ContractAnalyzer").then((m) => m.ContractAnalyzer),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 rounded-full border-2 border-navy border-t-transparent animate-spin" />
      </div>
    ),
  }
);

export default async function SavedAnalysisPage({ params }: { params: { id: string } }) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    redirect("/login");
  }

  const id = Number(params.id);
  if (!Number.isInteger(id)) notFound();

  // user_id까지 조건에 넣어서 다른 사용자의 id를 URL로 추측해 넣어도 남의 결과를 못 연다.
  const { rows } = await db.query(
    `SELECT title, result FROM analyses WHERE id = $1 AND user_id = $2`,
    [id, session.user.id]
  );
  if (rows.length === 0) notFound();

  const title = rows[0].title as string;
  const result = rows[0].result as FullAnalyzeResult;

  return (
    <AnalyzeShell title={title} description="내 분석 히스토리에 저장된 결과입니다.">
      <ContractAnalyzer initialResult={result} initialTitle={title} />
    </AnalyzeShell>
  );
}
