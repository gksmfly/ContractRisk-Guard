// frontend/components/sections/ProductPreviewSection.tsx
import { AlertTriangle, Scale } from "lucide-react";

// 점은 두 상태뿐이다 — 확인이 필요한 조항(seal)과 지목된 조가 없는 조항(회색).
// ochre/forest 3색은 위험도 3단계 시절의 잔재라 쓰지 않는다(2026-08-31 이후 등급 없음).
const SIDEBAR = [
  { label: "제1조 (해지권 부여)", dot: "bg-seal" },
  { label: "제2조 (책임 제한)", dot: "bg-seal" },
  { label: "제3조 (환불 정책)", dot: "bg-slate-300" },
  { label: "제4조 (개인정보 처리)", dot: "bg-slate-300" },
];

// 실제 /analyze 결과 화면의 이진 배지·법령 인용 스타일을 그대로 재현한 정적 목업.
// ExamplesSection의 "즉시 해지권 부여" 사례와 동일한 데이터를 재사용해 실제 화면과 어긋나지 않게 한다.
// **실제 화면이 안 내는 것을 여기서 내지 말 것** — 위험도 등급과 신뢰도 수치가 그것이다.
export function ProductPreviewSection() {
  return (
    <section aria-labelledby="preview-heading" className="px-4 pb-20 -mt-6 relative z-10">
      <h2 id="preview-heading" className="sr-only">
        실제 분석 화면 미리보기
      </h2>
      <div className="max-w-4xl mx-auto">
        <div className="rounded-2xl border border-slate-200 shadow-2xl shadow-slate-900/10 overflow-hidden bg-white">
          {/* Browser chrome */}
          <div className="flex items-center gap-2 px-4 py-3 bg-slate-50 border-b border-slate-200">
            <div className="flex gap-1.5" aria-hidden>
              <span className="w-2.5 h-2.5 rounded-full bg-red-300" />
              <span className="w-2.5 h-2.5 rounded-full bg-amber-300" />
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-300" />
            </div>
            <div className="flex-1 flex justify-center">
              <span className="text-xs text-slate-400 bg-white border border-slate-200 rounded-full px-4 py-1">
verilex.app/analyze
              </span>
            </div>
          </div>

          <div className="flex flex-col md:flex-row">
            {/* Sidebar */}
            <div className="md:w-56 shrink-0 border-b md:border-b-0 md:border-r border-slate-100 p-3 flex md:flex-col gap-1 overflow-x-auto bg-slate-50/50">
              {SIDEBAR.map((c, i) => (
                <div
                  key={c.label}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs whitespace-nowrap shrink-0 ${
                    i === 0
                      ? "bg-white shadow-sm border border-slate-200 font-medium text-slate-900"
                      : "text-slate-500"
                  }`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${c.dot}`} aria-hidden />
                  {c.label}
                </div>
              ))}
            </div>

            {/* Detail */}
            <div className="flex-1 p-6 space-y-4 text-left">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <span className="inline-flex items-center gap-1.5 text-xs font-semibold bg-seal-soft text-seal border border-seal/20 rounded-full px-3 py-1">
                  <AlertTriangle className="h-3 w-3" aria-hidden /> 확인 필요
                </span>
                <span className="text-[10px] text-slate-400 font-mono">제9조 · 제7조 관련</span>
              </div>

              <p className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-sm text-slate-700 leading-relaxed">
                회사는 이용자에게{" "}
                <mark className="text-seal font-semibold underline decoration-wavy decoration-seal/70 underline-offset-4 not-italic bg-transparent">
                  사전 통지 없이
                </mark>{" "}
                언제든지 서비스 이용 계약을{" "}
                <mark className="text-seal font-semibold underline decoration-wavy decoration-seal/70 underline-offset-4 not-italic bg-transparent">
                  즉시 해지
                </mark>
                하거나 서비스 제공을 중단할 수 있으며, 이로 인한 손해에 대해{" "}
                <mark className="text-seal font-semibold underline decoration-wavy decoration-seal/70 underline-offset-4 not-italic bg-transparent">
                  어떠한 책임도 지지 않습니다
                </mark>
                .
              </p>

              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <Scale className="h-3 w-3" aria-hidden />
                  <span>확인해 볼 조문</span>
                </div>
                <div className="flex items-start gap-2 bg-navy-soft/40 border border-navy/10 rounded-lg p-2.5 text-xs">
                  <span className="text-navy font-mono font-semibold shrink-0">약관규제법 §9</span>
                  <span className="text-slate-600">사업자에게 법률에 없는 해지권을 부여해 고객에게 부당한 불이익을 주는 조항</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        <p className="text-center text-xs text-slate-400 mt-4">
          실제 분석 화면을 그대로 재현한 미리보기입니다.
        </p>
      </div>
    </section>
  );
}
