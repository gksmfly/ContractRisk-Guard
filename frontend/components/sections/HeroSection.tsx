// frontend/components/sections/HeroSection.tsx
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Pause, Play } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { CountUp } from "@/components/CountUp";

const STATS = [
  { value: "6단계", label: "AI 파이프라인 검증" },
  { value: "4.98:1↑", label: "WCAG AA 대비 보장" },
  { value: "법령 원문", label: "직접 인용 근거 제시" },
];

const SLIDES = [
  {
    badge: "공정거래위원회 공인 약관 데이터 기반",
    heading: ["약관 속 불리한 조항,", "AI가 법령 근거로 잡아냅니다"],
    sub: "약관규제법 조문과 판례를 직접 인용해 판단 근거를 보여주는 유일한 계약서 리스크 분석 서비스",
  },
  {
    badge: "개인 소비자를 위한 30초 진단",
    heading: ["가입하기 전에,", "이 약관 위험한지 먼저 확인하세요"],
    sub: "구독 해지·환불 조항처럼 자주 분쟁이 생기는 부분만 골라 위험도와 근거를 바로 보여드립니다",
  },
  {
    badge: "기업 법무팀을 위한 감사 수준 검토",
    heading: ["자사 약관 전체를,", "AI 파이프라인으로 사전 검토하세요"],
    sub: "조항 단위 위험도 분류부터 법령·판례 근거, PDF 감사 리포트까지 한 번에 처리합니다",
  },
];

const SLIDE_INTERVAL = 5000;

export function HeroSection() {
  const [index, setIndex] = useState(0);
  const [fading, setFading] = useState(false);
  const [hovering, setHovering] = useState(false);
  const [manuallyPaused, setManuallyPaused] = useState(false);
  const paused = hovering || manuallyPaused;

  useEffect(() => {
    if (paused) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const id = setInterval(() => {
      setFading(true);
      setTimeout(() => {
        setIndex((i) => (i + 1) % SLIDES.length);
        setFading(false);
      }, 300);
    }, SLIDE_INTERVAL);
    return () => clearInterval(id);
  }, [paused]);

  const slide = SLIDES[index];

  return (
    <section
      id="hero"
      aria-labelledby="hero-heading"
      className="relative min-h-[92vh] flex flex-col items-center justify-center px-4 pt-24 pb-20"
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
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
        <div
          aria-live="polite"
          aria-atomic="true"
          className={`space-y-8 transition-opacity duration-300 ease-out ${
            fading ? "opacity-0" : "opacity-100"
          }`}
        >
          {/* Badge */}
          <div className="flex justify-center">
            <Badge className="bg-slate-50 text-slate-600 border border-slate-200 px-4 py-1.5 text-xs font-medium rounded-full">
              {slide.badge}
            </Badge>
          </div>

          {/* Headline */}
          <div className="space-y-5">
            <h1
              id="hero-heading"
              className="text-4xl md:text-5xl font-bold tracking-normal leading-[1.45] text-slate-900"
            >
              {slide.heading.map((line) => (
                <span className="block" key={line}>
                  {line}
                </span>
              ))}
            </h1>
            <p className="text-slate-500 text-lg leading-relaxed">{slide.sub}</p>
          </div>
        </div>

        {/* Slide indicators */}
        <div className="flex items-center justify-center gap-3">
          <div className="flex gap-2" role="tablist" aria-label="히어로 메시지 전환">
            {SLIDES.map((s, i) => (
              <button
                key={s.badge}
                role="tab"
                aria-selected={i === index}
                aria-label={`${i + 1}번째 메시지 보기`}
                onClick={() => {
                  setFading(true);
                  setTimeout(() => {
                    setIndex(i);
                    setFading(false);
                  }, 300);
                }}
                className={`h-1.5 rounded-full transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2 ${
                  i === index ? "w-6 bg-navy" : "w-1.5 bg-slate-300 hover:bg-slate-400"
                }`}
              />
            ))}
          </div>
          <button
            onClick={() => setManuallyPaused((v) => !v)}
            aria-pressed={manuallyPaused}
            aria-label={manuallyPaused ? "메시지 자동 전환 재생" : "메시지 자동 전환 일시정지"}
            className="p-1 rounded-full text-slate-400 hover:text-navy transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
          >
            {manuallyPaused ? <Play className="h-3 w-3" aria-hidden /> : <Pause className="h-3 w-3" aria-hidden />}
          </button>
        </div>

        {/* CTA buttons */}
        <div className="pt-2 flex flex-col sm:flex-row gap-3 justify-center items-center">
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 px-8 py-3.5 bg-navy hover:opacity-90 text-white rounded-xl font-semibold transition-opacity text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
          >
            약관 분석 무료로 시작하기
            <span aria-hidden>→</span>
          </Link>
          <Link
            href="/login"
            className="inline-flex items-center gap-2 px-8 py-3.5 bg-white hover:bg-slate-50 text-slate-700 rounded-xl font-medium transition-colors text-sm border border-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
          >
            기업 도입 문의
          </Link>
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
