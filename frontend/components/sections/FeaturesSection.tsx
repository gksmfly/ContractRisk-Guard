// frontend/components/sections/FeaturesSection.tsx
import { Card, CardContent } from "@/components/ui/card";
import { Scale, Layers, Grid3x3, Highlighter } from "lucide-react";

const FEATURES = [
  {
    icon: Scale,
    iconColor: "text-navy",
    iconBg: "bg-navy-soft",
    title: "법령 원문 직접 인용",
    desc: "약관규제법 제9조 등 실제 조문을 인용해 판단 근거를 보여줍니다.",
  },
  {
    icon: Layers,
    iconColor: "text-seal",
    iconBg: "bg-seal-soft",
    title: "6단계 AI 파이프라인",
    desc: "Analysis → Retrieval → Evidence → Judgment → Red-team → Verification",
  },
  {
    icon: Grid3x3,
    iconColor: "text-forest",
    iconBg: "bg-forest-soft",
    title: "위험도 미니맵",
    desc: "전체 약관을 고위험/중위험/저위험 색상 블록으로 한눈에",
  },
  {
    icon: Highlighter,
    iconColor: "text-ochre",
    iconBg: "bg-ochre-soft",
    title: "근거 하이라이트 연결",
    desc: "판단 결과에서 원문 보기 클릭 시 해당 조항으로 스크롤 + 플래시",
  },
];

export function FeaturesSection() {
  return (
    <section
      id="features"
      aria-labelledby="features-heading"
      className="py-24 px-4 bg-slate-50"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-14 space-y-3">
          <p className="text-navy text-xs font-semibold tracking-widest uppercase">
            검증된 판단 장치
          </p>
          <h2
            id="features-heading"
            className="text-3xl md:text-4xl font-bold text-slate-900"
          >
            &ldquo;예쁘다&rdquo;가 아니라
            <br />
            &ldquo;이거 진짜 근거 있네&rdquo;가 이겨야 합니다
          </h2>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map((f) => (
            <Card
              key={f.title}
              className="bg-white border border-slate-200 hover:border-slate-300 hover:shadow-md transition-all"
            >
              <CardContent className="p-5 space-y-3">
                <div className={`inline-flex p-2.5 rounded-lg ${f.iconBg}`}>
                  <f.icon className={`h-4 w-4 ${f.iconColor}`} aria-hidden />
                </div>
                <h3 className="font-semibold text-slate-900">{f.title}</h3>
                <p className="text-slate-500 text-sm leading-relaxed">{f.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
