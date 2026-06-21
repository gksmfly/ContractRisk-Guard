// frontend/components/sections/FeaturesSection.tsx
import { Card, CardContent } from "@/components/ui/card";
import {
  AlertTriangle,
  Scale,
  BookOpen,
  HighlighterIcon,
  ShieldCheck,
  FileSearch,
} from "lucide-react";

const FEATURES = [
  {
    icon: AlertTriangle,
    iconColor: "text-red-500",
    iconBg: "bg-red-50",
    title: "해지 조항 탐지",
    desc: "사전 통지 없는 일방 해지, 즉시 해지권 부여 등 약관규제법 §9 위반 패턴을 자동으로 식별합니다.",
    detail: "약관규제법 §9 · 민법 §543~553",
  },
  {
    icon: Scale,
    iconColor: "text-amber-500",
    iconBg: "bg-amber-50",
    title: "책임제한 조항 탐지",
    desc: "포괄적 면책, 손해배상 배제 등 사업자 책임을 부당하게 제한하는 조항을 탐지합니다.",
    detail: "약관규제법 §7 · 민법 §750~766",
  },
  {
    icon: ShieldCheck,
    iconColor: "text-emerald-500",
    iconBg: "bg-emerald-50",
    title: "3단계 위험도 분류",
    desc: "High / Medium / Low 세 단계로 리스크 수준을 분류하고, 즉각적인 대응 우선순위를 제시합니다.",
    detail: "공정위 시정조치 2,488건 기반",
  },
  {
    icon: HighlighterIcon,
    iconColor: "text-blue-500",
    iconBg: "bg-blue-50",
    title: "위험 문구 하이라이트",
    desc: "원문에서 위험한 표현을 직접 하이라이트하여 어떤 부분이 문제인지 즉시 확인할 수 있습니다.",
    detail: "evidence_span 추출 기반",
  },
  {
    icon: BookOpen,
    iconColor: "text-violet-500",
    iconBg: "bg-violet-50",
    title: "법령 근거 자동 연결",
    desc: "분석 결과마다 관련 법령 조문과 판례를 연결하여 법적 판단의 근거를 명확하게 제시합니다.",
    detail: "법령 16건 · 해석례 26건 · 판례 1,995건",
  },
  {
    icon: FileSearch,
    iconColor: "text-sky-500",
    iconBg: "bg-sky-50",
    title: "판단 일관성 검증",
    desc: "GPT-4o와 KoELECTRA가 서로 교차 검증하여 분석 결과의 신뢰도를 보장합니다.",
    detail: "Forward-Backward 일치 확인",
  },
];

export function FeaturesSection() {
  return (
    <section
      id="features"
      aria-labelledby="features-heading"
      className="py-24 px-4 bg-white"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-14 space-y-3">
          <p className="text-blue-600 text-xs font-semibold tracking-widest uppercase">
            기능
          </p>
          <h2
            id="features-heading"
            className="text-3xl md:text-4xl font-bold text-slate-900"
          >
            한 번의 분석으로 확인하는 것들
          </h2>
          <p className="text-slate-500 max-w-xl mx-auto">
            조항을 붙여넣으면 아래 6가지를 한꺼번에 제공합니다.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f) => (
            <Card
              key={f.title}
              className="bg-white border border-slate-200 hover:border-slate-300 hover:shadow-md transition-all"
            >
              <CardContent className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className={`p-2 rounded-lg ${f.iconBg}`}>
                    <f.icon className={`h-4 w-4 ${f.iconColor}`} aria-hidden />
                  </div>
                  <span className="text-[10px] font-mono text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                    {f.detail}
                  </span>
                </div>
                <h3 className="font-semibold text-slate-900">{f.title}</h3>
                <p className="text-slate-600 text-sm leading-relaxed">{f.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
