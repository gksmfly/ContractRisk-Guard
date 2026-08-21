// frontend/components/sections/HowToUseSection.tsx
import Link from "next/link";
import { Upload, Cpu, FileText } from "lucide-react";

const STEPS = [
  {
    num: "01",
    icon: Upload,
    iconColor: "text-navy",
    borderColor: "border-navy/20",
    bgColor: "bg-navy-soft",
    stepBadgeBg: "bg-navy-soft border-navy/20 text-navy",
    iconWrap: "bg-navy-soft",
    title: "계약서를 업로드하세요",
    desc: "계약서 파일(.txt)을 드래그하거나, 텍스트를 직접 붙여넣으세요. AI가 조항을 자동으로 분리합니다.",
    chips: [],
  },
  {
    num: "02",
    icon: Cpu,
    iconColor: "text-violet-600",
    borderColor: "border-violet-200",
    bgColor: "bg-violet-50/60",
    stepBadgeBg: "bg-violet-100 border-violet-200 text-violet-700",
    iconWrap: "bg-violet-100",
    title: "AI가 조항별로 분석합니다",
    desc: "공정위 시정조치 2,488건을 학습한 AI가 각 조항의 유형과 위험도를 판단합니다.",
    chips: ["6단계 AI 파이프라인", "위험도 미니맵"],
  },
  {
    num: "03",
    icon: FileText,
    iconColor: "text-emerald-600",
    borderColor: "border-emerald-200",
    bgColor: "bg-emerald-50/60",
    stepBadgeBg: "bg-emerald-100 border-emerald-200 text-emerald-700",
    iconWrap: "bg-emerald-100",
    title: "리스크 레포트를 확인하세요",
    desc: "위험도·분류·관련 법령·위험 문구 하이라이트가 담긴 조항별 분석 결과를 바로 확인합니다.",
    chips: ["법령 원문 직접 인용", "근거 하이라이트 연결"],
  },
];

export function HowToUseSection() {
  return (
    <section
      id="how-to-use"
      aria-labelledby="how-to-use-heading"
      className="py-24 px-4 bg-slate-50"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-14 space-y-3">
          <p className="text-navy text-xs font-semibold tracking-widest uppercase">
            이용 방법
          </p>
          <h2
            id="how-to-use-heading"
            className="text-3xl md:text-4xl font-bold text-slate-900"
          >
            3단계로 끝납니다
          </h2>
          <p className="text-slate-500 max-w-xl mx-auto">
            &ldquo;예쁘다&rdquo;가 아니라 &ldquo;이거 진짜 근거 있네&rdquo;가 이겨야 합니다 — 회원가입도 설치도 필요 없습니다.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 relative">
          {/* Connector lines (desktop) */}
          <div className="hidden md:block absolute top-10 left-[calc(33.33%+1rem)] right-[calc(33.33%+1rem)] h-px bg-gradient-to-r from-navy/20 via-navy/40 to-navy/20" />

          {STEPS.map((step) => (
            <div
              key={step.num}
              className={`relative border ${step.borderColor} ${step.bgColor} bg-white rounded-2xl p-6 space-y-4 shadow-sm`}
            >
              <div className="flex items-center justify-between">
                <span
                  className={`text-xs font-mono font-bold px-2.5 py-1 rounded-full border ${step.stepBadgeBg}`}
                >
                  STEP {step.num}
                </span>
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${step.iconWrap}`}>
                  <step.icon
                    className={`h-5 w-5 ${step.iconColor}`}
                    aria-hidden
                  />
                </div>
              </div>

              <h3 className="font-bold text-slate-900 text-lg leading-snug">
                {step.title}
              </h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                {step.desc}
              </p>

              {step.chips.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {step.chips.map((chip) => (
                    <span
                      key={chip}
                      className="text-[11px] font-mono bg-slate-100 text-slate-500 rounded-md px-2 py-1"
                    >
                      {chip}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="mt-10 text-center">
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 px-6 py-3 bg-navy hover:opacity-90 text-white rounded-xl font-medium transition-colors text-sm shadow-sm"
          >
            지금 무료로 분석해보기
          </Link>
        </div>
      </div>
    </section>
  );
}
