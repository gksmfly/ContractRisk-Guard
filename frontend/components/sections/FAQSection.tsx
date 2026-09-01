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
    q: "AI는 어떤 기준으로 판단하나요?",
    // Red-team 단계는 지금 비활성이다(이웃 테이블이 옛 라벨로 적재돼 있어 아무것도 발동하지
    // 않는다). 동작하지 않는 것을 FAQ에 적어두면 그게 곧 허위 설명이라 뺐다.
    a: "공정거래위원회가 실제로 시정조치를 내린 의결서 1,163건에서 추출한 조항을 학습한 분류 모델이 판단합니다. 함께 표시되는 조문은 검색 결과가 아니라 모델이 지목한 조의 법령 원문을 그대로 가져온 것이며, 판례는 참고 사례로만 제시합니다.",
  },
  {
    q: "위험도를 상·중·하로 알려주나요?",
    a: "아니요. 등급이나 확률 수치는 표시하지 않습니다. 저희가 근거를 갖고 말씀드릴 수 있는 것은 '이 조항을 확인해 보세요'까지이고, 어느 조에 걸리는지는 참고 정보로만 제시합니다. 검증하지 않은 등급을 보여드리면 오히려 '검토했고 문제없다'는 잘못된 안심을 드리게 되기 때문입니다.",
  },
  {
    q: "어떤 조항까지 분석할 수 있나요?",
    a: "약관규제법 제6~14조(일반원칙·면책·손해배상액·해지·급부·권익보호·의사표시·대리인책임·소송) 위반 소지가 있는 조항을 찾습니다. 한 번에 최대 60개 조항까지 처리하며, 이 범위 안에서도 모든 조항을 찾아내지는 못합니다.",
  },
  {
    q: "입력한 계약서 내용은 저장되나요?",
    a: "분석 자체는 저장하지 않고 처리 후 즉시 폐기합니다. 결과 화면에서 직접 '저장하기'를 누른 경우에만 그 분석 결과(원문 포함)가 내 계정의 히스토리에 남습니다.",
  },
  {
    q: "비용이 드나요? 회원가입이 필요한가요?",
    a: "이용 자체는 무료이지만, 안전한 이용을 위해 Google 계정으로 로그인이 필요합니다.",
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
