// frontend/components/sections/HeroSection.tsx
"use client";

import Link from "next/link";
import { ArrowRight, CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const BULLETS = [
  "해지·책임제한 조항 자동 분류",
  "약관규제법·민법 법령 근거 제시",
  "위험 문구 하이라이트",
];

export function HeroSection() {
  return (
    <section
      id="hero"
      aria-labelledby="hero-heading"
      className="relative min-h-[92vh] flex flex-col items-center justify-center px-4 pt-24 pb-20 overflow-hidden"
    >
      {/* Subtle gradient background */}
      <div className="absolute inset-0 bg-gradient-to-b from-blue-50/80 via-white to-white pointer-events-none" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[700px] h-[300px] bg-blue-100/60 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10 w-full max-w-2xl mx-auto text-center space-y-7">
        {/* Badge */}
        <div className="flex justify-center">
          <Badge className="bg-blue-50 text-blue-600 border border-blue-200 px-4 py-1.5 text-xs font-medium rounded-full">
            공정위 시정조치 2,488건 기반 · 약관규제법 §7, §9
          </Badge>
        </div>

        {/* Headline */}
        <div className="space-y-4">
          <h1
            id="hero-heading"
            className="text-4xl md:text-5xl font-extrabold tracking-tight leading-tight text-slate-900"
          >
            계약서 조항,
            <br />
            <span className="bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
              붙여넣기만 하세요
            </span>
          </h1>
          <p className="text-slate-600 text-lg leading-relaxed">
            해지·책임제한 조항의 법적 리스크를 AI가 10초 안에 분석하고
            <br className="hidden md:block" />
            <span className="text-slate-900 font-medium">법령 근거와 위험 문구를 함께 제시</span>합니다.
          </p>
        </div>

        {/* Bullet points */}
        <div className="flex flex-wrap justify-center gap-x-6 gap-y-2">
          {BULLETS.map((b) => (
            <div key={b} className="flex items-center gap-1.5 text-sm text-slate-600">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
              <span>{b}</span>
            </div>
          ))}
        </div>

        {/* CTA buttons */}
        <div className="pt-2 flex flex-col sm:flex-row gap-3 justify-center items-center">
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 px-8 py-3.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-semibold transition-colors text-sm shadow-lg shadow-blue-500/25"
          >
            계약서 분석 시작
            <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
          <a
            href="#examples"
            className="inline-flex items-center gap-2 px-8 py-3.5 bg-white hover:bg-slate-50 text-slate-700 rounded-xl font-medium transition-colors text-sm border border-slate-200"
          >
            분석 사례 보기
          </a>
        </div>

        {/* Trust indicators */}
        <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 pt-1">
          {["회원가입 불필요", "무료 이용", "텍스트 저장 안 함"].map((item) => (
            <span key={item} className="text-xs text-slate-400">{item}</span>
          ))}
        </div>

        <p className="text-xs text-slate-400">
          본 서비스의 분석 결과는 참고용이며 법적 조언을 대체하지 않습니다.
        </p>
      </div>
    </section>
  );
}
