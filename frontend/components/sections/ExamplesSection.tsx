// frontend/components/sections/ExamplesSection.tsx
"use client";

import { useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
  AlertTriangle,
  CheckCircle2,
  Shield,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

interface ExampleResult {
  domain: string;
  riskLevel: "High" | "Medium" | "Low";
  confidence: number;
  highlights: { text: string }[];
  legalBasis: { law: string; article: string; desc: string }[];
  summary: string;
}

interface Example {
  id: number;
  label: string;
  clause: string;
  result: ExampleResult;
}

const EXAMPLES: Example[] = [
  {
    id: 1,
    label: "즉시 해지권 부여",
    clause:
      "회사는 이용자에게 사전 통지 없이 언제든지 서비스 이용 계약을 즉시 해지하거나 서비스 제공을 중단할 수 있으며, 이로 인한 손해에 대해 어떠한 책임도 지지 않습니다.",
    result: {
      domain: "해지_조항",
      riskLevel: "High",
      confidence: 0.93,
      highlights: [
        { text: "사전 통지 없이" },
        { text: "즉시 해지" },
        { text: "어떠한 책임도 지지 않습니다" },
      ],
      legalBasis: [
        { law: "약관규제법", article: "§9", desc: "고객에게 부당하게 불리한 해제·해지권 부여 조항 무효" },
        { law: "민법", article: "§544", desc: "계약 해제 시 상당한 기간을 정한 최고 요건" },
      ],
      summary:
        "사전 통지 없는 즉시 해지권을 사업자에게 일방적으로 부여하고 이에 따른 손해 책임마저 면제하는 조항입니다. 약관규제법 §9에 따라 무효 처리될 가능성이 높습니다.",
    },
  },
  {
    id: 2,
    label: "포괄적 면책 조항",
    clause:
      "회사는 서비스 이용과 관련하여 발생한 이용자의 손해에 대하여 회사의 귀책사유 유무에 관계없이 일체의 배상 책임을 부담하지 아니합니다.",
    result: {
      domain: "책임제한_조항",
      riskLevel: "High",
      confidence: 0.91,
      highlights: [
        { text: "귀책사유 유무에 관계없이" },
        { text: "일체의 배상 책임을 부담하지 아니합니다" },
      ],
      legalBasis: [
        { law: "약관규제법", article: "§7", desc: "사업자 고의·과실로 인한 법률상 책임 배제 조항 무효" },
        { law: "민법", article: "§750", desc: "불법행위 손해배상 책임" },
      ],
      summary:
        "사업자의 귀책사유(고의·과실)와 무관하게 모든 손해배상을 면제하는 포괄적 면책 조항입니다. 약관규제법 §7에 의해 무효가 됩니다.",
    },
  },
  {
    id: 3,
    label: "표준 해지 조항 (정상)",
    clause:
      "계약 당사자는 상대방에게 30일 이상의 사전 서면 통지를 통해 본 계약을 해지할 수 있으며, 해지 시 미이행 채무에 대한 정산은 쌍방 합의에 따릅니다.",
    result: {
      domain: "해지_조항",
      riskLevel: "Low",
      confidence: 0.88,
      highlights: [],
      legalBasis: [
        { law: "민법", article: "§543", desc: "해지권 행사의 일반 원칙" },
      ],
      summary:
        "30일 사전 서면 통지를 요구하는 표준적인 해지 조항으로, 양 당사자의 권리를 균형 있게 보호합니다. 법적 리스크가 낮습니다.",
    },
  },
];

// /analyze 실제 결과 화면(ContractAnalyzer.tsx의 RISK_CONFIG)과 동일한 seal/ochre/forest 토큰을 사용한다.
const RISK_CONFIG = {
  High: {
    badgeClass: "bg-seal-soft text-seal border-seal/20",
    icon: AlertTriangle,
    iconClass: "text-seal",
    ringClass: "ring-seal/20",
    metricText: "text-seal",
    label: "고위험",
  },
  Medium: {
    badgeClass: "bg-ochre-soft text-ochre border-ochre/20",
    icon: AlertTriangle,
    iconClass: "text-ochre",
    ringClass: "ring-ochre/20",
    metricText: "text-ochre",
    label: "중위험",
  },
  Low: {
    badgeClass: "bg-forest-soft text-forest border-forest/20",
    icon: CheckCircle2,
    iconClass: "text-forest",
    ringClass: "ring-forest/20",
    metricText: "text-forest",
    label: "저위험",
  },
};

function HighlightedClause({
  text,
  highlights,
}: {
  text: string;
  highlights: { text: string }[];
}) {
  if (!highlights.length) {
    return <span className="text-slate-700 text-sm leading-relaxed">{text}</span>;
  }

  const parts: { t: string; hi: boolean }[] = [];
  let remaining = text;

  for (const h of highlights) {
    const idx = remaining.indexOf(h.text);
    if (idx === -1) continue;
    if (idx > 0) parts.push({ t: remaining.slice(0, idx), hi: false });
    parts.push({ t: h.text, hi: true });
    remaining = remaining.slice(idx + h.text.length);
  }
  if (remaining) parts.push({ t: remaining, hi: false });

  if (!parts.length) return <span className="text-slate-700 text-sm leading-relaxed">{text}</span>;

  return (
    <span className="text-sm leading-relaxed">
      {parts.map((p, i) =>
        p.hi ? (
          <mark
            key={i}
            className="text-seal font-semibold underline decoration-wavy decoration-seal/70 underline-offset-4 rounded px-0.5 not-italic bg-transparent"
          >
            {p.t}
          </mark>
        ) : (
          <span key={i} className="text-slate-700">{p.t}</span>
        )
      )}
    </span>
  );
}

function ExampleCard({ ex }: { ex: Example }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = RISK_CONFIG[ex.result.riskLevel];
  const RiskIcon = cfg.icon;

  return (
    <article className="bg-white border border-slate-200 rounded-2xl overflow-hidden hover:border-slate-300 hover:shadow-md transition-all">
      {/* Header */}
      <div className="p-5 flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg bg-slate-100 ring-2 ${cfg.ringClass}`}>
            <RiskIcon className={`h-4 w-4 ${cfg.iconClass}`} aria-hidden />
          </div>
          <div>
            <span className="text-xs text-slate-400">예시 {ex.id}</span>
            <h3 className="font-semibold text-slate-900 text-sm">{ex.label}</h3>
          </div>
        </div>
        <Badge
          className={`${cfg.badgeClass} border text-xs shrink-0`}
          aria-label={`위험도: ${cfg.label}`}
        >
          {cfg.label}
        </Badge>
      </div>

      {/* Clause */}
      <div className="px-5 pb-4">
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
          <HighlightedClause
            text={ex.clause}
            highlights={ex.result.highlights}
          />
        </div>
      </div>

      {/* Toggle */}
      <button
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="w-full px-5 py-3 flex items-center justify-between text-xs text-slate-500 hover:text-slate-700 border-t border-slate-200 transition-colors"
      >
        <span>분석 결과 {expanded ? "접기" : "보기"}</span>
        {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
      </button>

      {/* Result detail */}
      {expanded && (
        <div className="px-5 pb-5 space-y-4 border-t border-slate-100">
          {/* Metrics row */}
          <div className="flex gap-3 pt-4">
            <div className="flex-1 bg-slate-50 rounded-lg p-3 text-center border border-slate-100">
              <p className="text-[10px] text-slate-400 mb-0.5">분류</p>
              <p className="text-sm font-semibold text-slate-900">{ex.result.domain}</p>
            </div>
            <div className="flex-1 bg-slate-50 rounded-lg p-3 text-center border border-slate-100">
              <p className="text-[10px] text-slate-400 mb-0.5">위험도</p>
              <p className={`text-sm font-bold ${cfg.metricText}`}>{cfg.label}</p>
            </div>
            <div className="flex-1 bg-slate-50 rounded-lg p-3 text-center border border-slate-100">
              <p className="text-[10px] text-slate-400 mb-0.5">신뢰도</p>
              <p className="text-sm font-semibold text-slate-900">
                {Math.round(ex.result.confidence * 100)}%
              </p>
            </div>
          </div>

          {/* Legal basis */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5 text-xs text-slate-500">
              <Shield className="h-3 w-3" aria-hidden />
              <span>적용 법령</span>
            </div>
            {ex.result.legalBasis.map((lb) => (
              <div
                key={lb.article}
                className="flex items-start gap-2 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs"
              >
                <span className="text-navy font-mono font-semibold shrink-0">
                  {lb.law} {lb.article}
                </span>
                <span className="text-slate-600">{lb.desc}</span>
              </div>
            ))}
          </div>

          {/* Summary */}
          <p className="text-xs text-slate-600 leading-relaxed bg-navy-soft border border-navy/10 rounded-lg p-3">
            {ex.result.summary}
          </p>
        </div>
      )}
    </article>
  );
}

export function ExamplesSection() {
  const trackRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);

  const scrollToIndex = (i: number) => {
    const track = trackRef.current;
    if (!track) return;
    const card = track.children[i] as HTMLElement | undefined;
    card?.scrollIntoView({ behavior: "smooth", inline: "start", block: "nearest" });
  };

  const handleScroll = () => {
    const track = trackRef.current;
    if (!track) return;
    const cardWidth = (track.firstElementChild as HTMLElement | null)?.offsetWidth ?? 1;
    const gap = 20; // gap-5
    setActive(Math.round(track.scrollLeft / (cardWidth + gap)));
  };

  return (
    <section
      id="examples"
      aria-labelledby="examples-heading"
      className="py-24 px-4 bg-white"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-14 space-y-3">
          <p className="text-navy text-xs font-semibold tracking-widest uppercase">
            분석 사례
          </p>
          <h2
            id="examples-heading"
            className="text-3xl md:text-4xl font-bold text-slate-900"
          >
            이런 조항들을 탐지합니다
          </h2>
          <p className="text-slate-500 max-w-xl mx-auto">
            공정위에 실제 시정조치된 유형을 기반으로 학습한 결과입니다.
          </p>
        </div>

        {/* Carousel */}
        <div className="relative">
          <div
            ref={trackRef}
            onScroll={handleScroll}
            className="flex gap-5 overflow-x-auto snap-x snap-mandatory scroll-smooth pb-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          >
            {EXAMPLES.map((ex) => (
              <div key={ex.id} className="snap-start shrink-0 w-full sm:w-[calc(50%-0.625rem)] md:w-[calc(33.333%-0.834rem)]">
                <ExampleCard ex={ex} />
              </div>
            ))}
          </div>

          {/* Arrow controls */}
          <button
            type="button"
            onClick={() => scrollToIndex(Math.max(active - 1, 0))}
            disabled={active === 0}
            aria-label="이전 사례"
            className="hidden md:flex absolute -left-4 top-1/2 -translate-y-1/2 items-center justify-center w-9 h-9 rounded-full bg-white border border-slate-200 shadow-md text-slate-500 hover:text-navy hover:border-navy/30 disabled:opacity-0 disabled:pointer-events-none transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
          >
            <ChevronLeft className="h-4 w-4" aria-hidden />
          </button>
          <button
            type="button"
            onClick={() => scrollToIndex(Math.min(active + 1, EXAMPLES.length - 1))}
            disabled={active === EXAMPLES.length - 1}
            aria-label="다음 사례"
            className="hidden md:flex absolute -right-4 top-1/2 -translate-y-1/2 items-center justify-center w-9 h-9 rounded-full bg-white border border-slate-200 shadow-md text-slate-500 hover:text-navy hover:border-navy/30 disabled:opacity-0 disabled:pointer-events-none transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
          >
            <ChevronRight className="h-4 w-4" aria-hidden />
          </button>
        </div>

        {/* Dot indicators (mobile) */}
        <div className="flex md:hidden justify-center gap-1.5 mt-4">
          {EXAMPLES.map((ex, i) => (
            <button
              key={ex.id}
              onClick={() => scrollToIndex(i)}
              aria-label={`${i + 1}번째 사례 보기`}
              className={`h-1.5 rounded-full transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2 ${
                i === active ? "w-6 bg-navy" : "w-1.5 bg-slate-300"
              }`}
            />
          ))}
        </div>

        <p className="text-center text-xs text-slate-400 mt-6">
          위 예시는 실제 공정위 시정조치 유형을 참고하여 구성된 설명용 사례입니다.
        </p>
      </div>
    </section>
  );
}
