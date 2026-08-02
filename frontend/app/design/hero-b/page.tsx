// frontend/app/design/hero-b/page.tsx
// 비교용 시안 — 실제 랜딩(app/page.tsx)과 별개. 채택 여부 결정 전까지 여기서만 확인한다.
"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowRight } from "lucide-react";
import { Logo } from "@/components/Logo";

export default function HeroPreviewB() {
  const router = useRouter();
  const [text, setText] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    router.push("/analyze");
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="bg-navy text-white text-xs text-center py-2 px-4">
        비교용 시안입니다 — 실제 홈페이지가 아닙니다.{" "}
        <Link href="/" className="underline hover:text-navy-soft">실제 홈페이지 보기 →</Link>
      </div>

      {/* Full-bleed hero */}
      <section className="relative h-[85vh] min-h-[560px] overflow-hidden flex flex-col">
        {/* Background: 추상 그라디언트 + 방패/저울 워터마크 (실사 이미지 없이 브랜드 모티프로 구성) */}
        <div className="absolute inset-0 bg-navy" />
        <div className="absolute inset-0 opacity-[0.07] flex items-center justify-end pr-0">
          <Logo className="h-[140%] w-auto text-white translate-x-1/4" />
        </div>
        <div className="absolute top-1/3 left-1/4 w-96 h-96 bg-navy-soft/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-72 h-72 bg-seal/10 rounded-full blur-3xl" />

        {/* Header (transparent, over the image) */}
        <header className="relative z-10 max-w-6xl mx-auto w-full px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 group">
            <Logo className="h-5 w-5 text-white group-hover:text-navy-soft transition-colors" />
            <span className="font-bold text-white text-sm">Verilex</span>
          </Link>
          <nav className="hidden md:flex items-center gap-7 text-sm text-white/70">
            <a href="/#how-to-use" className="hover:text-white transition-colors">이용 방법</a>
            <a href="/#examples" className="hover:text-white transition-colors">분석 사례</a>
            <a href="/#trust" className="hover:text-white transition-colors">신뢰 근거</a>
          </nav>
          <Link href="/login" className="text-sm text-white/90 hover:text-white border border-white/30 rounded-lg px-4 py-1.5 transition-colors">
            로그인
          </Link>
        </header>

        {/* Center content */}
        <div className="relative z-10 flex-1 flex flex-col items-center justify-center px-6 text-center">
          <p className="text-white/60 text-xs font-medium tracking-widest uppercase mb-5">
            공정거래위원회 공인 약관 데이터 기반
          </p>
          <h1 className="text-3xl md:text-5xl font-bold text-white leading-[1.4] mb-8 max-w-3xl">
            약관 속 불리한 조항,
            <br />
            AI가 법령 근거로 잡아냅니다
          </h1>

          <form
            onSubmit={handleSubmit}
            className="w-full max-w-xl bg-white/95 backdrop-blur rounded-full shadow-2xl flex items-center pl-6 pr-2 py-2"
          >
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="약관 조항을 붙여넣고 위험도를 확인하세요"
              className="flex-1 bg-transparent outline-none text-sm text-slate-700 placeholder:text-slate-400"
            />
            <button
              type="submit"
              aria-label="계약서 분석 시작"
              className="shrink-0 w-10 h-10 rounded-full bg-navy hover:opacity-90 text-white flex items-center justify-center transition-opacity"
            >
              <ArrowRight className="h-4 w-4" aria-hidden />
            </button>
          </form>

          <p className="text-white/50 text-xs mt-5">
            회원가입 불필요 · 무료 이용 · 텍스트 저장 안 함
          </p>
        </div>
      </section>

      <div className="max-w-2xl mx-auto text-center py-10 px-4">
        <p className="text-sm text-slate-500">
          현재 실제 홈페이지는 텍스트 중심(카드+통계) 히어로를 쓰고 있습니다. 이 시안은 풀블리드 배경 +
          중앙 검색창 스타일로 만든 대안입니다 — 마음에 드시면 이 방향으로 교체해드릴게요.
        </p>
      </div>
    </div>
  );
}
