// frontend/components/sections/HeroSection.tsx
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { CountUp } from "@/components/CountUp";

const STATS = [
  { value: "6단계", label: "AI 파이프라인 검증" },
  { value: "4.98:1↑", label: "WCAG AA 대비 보장" },
  { value: "법령 원문", label: "직접 인용 근거 제시" },
];

export function HeroSection() {
  return (
    <section
      id="hero"
      aria-labelledby="hero-heading"
      className="relative min-h-[92vh] flex flex-col items-center justify-center px-4 pt-24 pb-20"
    >
      {/* 배경 그래픽 — 정보 전달용이 아닌 순수 장식, 스크린리더에서 숨김 */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden>
        <div className="absolute -top-24 -left-24 w-[28rem] h-[28rem] rounded-full bg-navy-soft/60 blur-3xl" />
        <div className="absolute top-1/3 -right-32 w-[26rem] h-[26rem] rounded-full bg-ochre-soft/70 blur-3xl" />
        <div className="absolute -bottom-32 left-1/4 w-[24rem] h-[24rem] rounded-full bg-forest-soft/60 blur-3xl" />
        <svg
          className="absolute inset-0 w-full h-full opacity-[0.15]"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <pattern id="hero-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M40 0H0V40" fill="none" stroke="#0f1e3d" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#hero-grid)" />
        </svg>
      </div>

      <div className="relative z-10 w-full max-w-2xl mx-auto text-center space-y-8">
        {/* Badge */}
        <div className="flex justify-center">
          <Badge className="bg-slate-50 text-slate-600 border border-slate-200 px-4 py-1.5 text-xs font-medium rounded-full">
            공정거래위원회 공인 약관 데이터 기반
          </Badge>
        </div>

        {/* Headline */}
        <div className="space-y-5">
          <h1
            id="hero-heading"
            className="text-4xl md:text-5xl font-bold tracking-normal leading-[1.45] text-slate-900"
          >
            <span className="block">약관 속 불리한 조항,</span>
            <span className="block">AI가 법령 근거로 잡아냅니다</span>
          </h1>
          <p className="text-slate-500 text-lg leading-relaxed">
            약관규제법 조문과 판례를 직접 인용해 판단 근거를 보여주는 유일한 계약서 리스크 분석 서비스
          </p>
        </div>

        {/* CTA */}
        <div className="pt-2">
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 px-8 py-3.5 bg-navy hover:opacity-90 text-white rounded-xl font-semibold transition-opacity text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
          >
            약관 분석 무료로 시작하기
            <span aria-hidden>→</span>
          </Link>
          <p className="text-slate-400 text-xs mt-3">로그인 없이 바로 시작 · 원하면 나중에 저장 가능</p>
        </div>

        {/* Stats */}
        <div className="pt-4 grid grid-cols-3 gap-4 max-w-md mx-auto">
          {STATS.map((s) => (
            <div key={s.label}>
              <p className="text-navy font-bold text-lg">
                <CountUp value={s.value} />
              </p>
              <p className="text-slate-400 text-xs mt-0.5">{s.label}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
