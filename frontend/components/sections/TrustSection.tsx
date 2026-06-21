// frontend/components/sections/TrustSection.tsx
import { Scale, BookOpen, Database, ShieldCheck } from "lucide-react";

const STATS = [
  {
    icon: ShieldCheck,
    iconColor: "text-blue-600",
    iconBg: "bg-blue-50",
    value: "2,488건",
    label: "학습 기반",
    desc: "공정위 시정조치 사례를 직접 학습한 Ground Truth 데이터",
  },
  {
    icon: Database,
    iconColor: "text-violet-600",
    iconBg: "bg-violet-50",
    value: "1,995건",
    label: "참조 판례",
    desc: "해지·책임제한 도메인으로 필터링된 실제 법원 판례",
  },
  {
    icon: BookOpen,
    iconColor: "text-emerald-600",
    iconBg: "bg-emerald-50",
    value: "42건",
    label: "법령·해석례",
    desc: "법령 16건 + 해석례 26건, 법적 근거 연결에 사용",
  },
  {
    icon: Scale,
    iconColor: "text-amber-600",
    iconBg: "bg-amber-50",
    value: "4개",
    label: "적용 법령",
    desc: "약관규제법 §7·§9, 민법 §543~553, §750~766",
  },
];

const COVER_LAWS = [
  { law: "약관규제법", articles: "§7", desc: "사업자 책임 부당 면제·제한 조항" },
  { law: "약관규제법", articles: "§9", desc: "부당한 해제·해지권 부여 조항" },
  { law: "민법", articles: "§543~553", desc: "계약 해제·해지의 요건 및 효과" },
  { law: "민법", articles: "§750~766", desc: "불법행위 손해배상 책임" },
];

export function TrustSection() {
  return (
    <section
      id="trust"
      aria-labelledby="trust-heading"
      className="py-24 px-4 bg-slate-50"
    >
      <div className="max-w-5xl mx-auto space-y-14">
        {/* Header */}
        <div className="text-center space-y-3">
          <p className="text-blue-600 text-xs font-semibold tracking-widest uppercase">
            신뢰 근거
          </p>
          <h2
            id="trust-heading"
            className="text-3xl md:text-4xl font-bold text-slate-900"
          >
            어디서 온 AI인가요?
          </h2>
          <p className="text-slate-500 max-w-xl mx-auto">
            공개된 법령 데이터와 공정위 공식 자료만을 사용하여 학습했습니다.
          </p>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {STATS.map((s) => (
            <div
              key={s.label}
              className="bg-white border border-slate-200 rounded-2xl p-5 space-y-3 shadow-sm"
            >
              <div className={`p-2 rounded-lg ${s.iconBg} w-fit`}>
                <s.icon className={`h-4 w-4 ${s.iconColor}`} aria-hidden />
              </div>
              <div>
                <p className="text-2xl font-extrabold text-slate-900">{s.value}</p>
                <p className={`text-xs font-semibold ${s.iconColor} mt-0.5`}>{s.label}</p>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">{s.desc}</p>
            </div>
          ))}
        </div>

        {/* Legal coverage */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 md:p-8 shadow-sm">
          <h3 className="text-slate-900 font-semibold mb-5 flex items-center gap-2">
            <Scale className="h-4 w-4 text-blue-600" aria-hidden />
            분석 적용 법령
          </h3>
          <div className="grid md:grid-cols-2 gap-3">
            {COVER_LAWS.map((l) => (
              <div
                key={l.articles}
                className="flex items-start gap-3 bg-slate-50 border border-slate-200 rounded-xl p-4"
              >
                <div className="shrink-0 mt-0.5">
                  <span className="inline-flex items-center gap-1 text-[10px] font-mono bg-blue-50 text-blue-600 border border-blue-200 px-2 py-0.5 rounded">
                    {l.law} {l.articles}
                  </span>
                </div>
                <p className="text-slate-600 text-sm">{l.desc}</p>
              </div>
            ))}
          </div>

          <p className="mt-5 text-xs text-slate-400 leading-relaxed border-t border-slate-200 pt-4">
            학습 데이터 출처: 공정거래위원회 심결례 공개 자료, 국가법령정보센터 API (법령·판례·해석례),
            한국공정거래조정원 표준계약서. 개인정보 미포함.
          </p>
        </div>
      </div>
    </section>
  );
}
