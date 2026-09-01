// frontend/components/sections/FAQSection.tsx
"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

const FAQS: { q: string; a: string }[] = [
  {
    q: "이 서비스는 법적 효력이 있는 자문인가요?",
    a: "아니요. 참고용 리스크 진단 도구이며 법적 조언을 대체하지 않습니다. 실제 계약 체결 전에는 반드시 법률 전문가의 검토를 받으시기 바랍니다.",
  },
  {
    q: "AI는 어떤 기준으로 위험도를 판단하나요?",
    a: "공정거래위원회가 실제로 시정조치를 내린 사례 2,488건을 학습 데이터로 삼아 분류 모델이 1차 판단을 내리고, Red-team 에이전트가 유사 조항의 과거 판단과 비교해 편향 여부를 재검증합니다. 근거로 제시되는 법령·판례는 실제 조문·판결 원문에서 검색해 인용합니다.",
  },
  {
    q: "어떤 조항까지 분석할 수 있나요?",
    a: "약관규제법 제6~14조(일반원칙·면책·손해배상액·해지·급부·권익보호·의사표시·대리인책임·소송) 위반 소지가 있는 조항을 찾습니다. 한 번에 최대 60개 조항까지 처리하며, 이 범위 안에서도 모든 조항을 찾아내지는 못합니다.",
  },
  {
    q: "입력한 계약서 내용은 저장되나요?",
    a: "아니요. 분석 처리에만 사용되며 서버에 별도로 저장하지 않습니다.",
  },
  {
    q: "비용이 드나요? 회원가입이 필요한가요?",
    a: "무료이며 회원가입 없이 바로 이용할 수 있습니다.",
  },
  {
    q: "PDF 파일도 분석할 수 있나요?",
    a: "네, PDF와 TXT 파일 업로드 및 텍스트 직접 붙여넣기를 지원합니다. 파일당 최대 10MB까지 가능합니다.",
  },
];

function FaqItem({ q, a, defaultOpen }: { q: string; a: string; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(!!defaultOpen);

  return (
    <div className="border border-slate-200 rounded-xl bg-white overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full flex items-start gap-3 px-5 py-4 text-left rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
      >
        <span className="mt-0.5 shrink-0 w-5 h-5 rounded-full bg-navy-soft text-navy text-xs font-bold flex items-center justify-center">
          Q
        </span>
        <span className="flex-1 text-sm font-medium text-slate-900">{q}</span>
        <ChevronDown
          className={`h-4 w-4 text-slate-400 shrink-0 mt-0.5 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
          aria-hidden
        />
      </button>
      <div
        className="grid transition-[grid-template-rows] duration-200 ease-out"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="flex items-start gap-3 px-5 pb-4">
            <span className="mt-0.5 shrink-0 w-5 h-5 rounded-full bg-slate-100 text-slate-500 text-xs font-bold flex items-center justify-center">
              A
            </span>
            <p className="flex-1 text-sm text-slate-600 leading-relaxed">{a}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export function FAQSection() {
  return (
    <section id="faq" aria-labelledby="faq-heading" className="py-24 px-4 bg-white">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-10 space-y-3">
          <p className="text-navy text-xs font-semibold tracking-widest uppercase">
            자주 묻는 질문
          </p>
          <h2 id="faq-heading" className="text-3xl md:text-4xl font-bold text-slate-900">
            궁금한 점을 확인하세요
          </h2>
        </div>

        <div className="space-y-3">
          {FAQS.map((f, i) => (
            <FaqItem key={f.q} q={f.q} a={f.a} defaultOpen={i === 0} />
          ))}
        </div>
      </div>
    </section>
  );
}
