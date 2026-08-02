// frontend/components/sections/AudienceSection.tsx
import Link from "next/link";

// 단일 문서를 빠르게 확인한다는 의미의 라인아트 일러스트 (B2C)
function QuickCheckIllustration() {
  return (
    <svg viewBox="0 0 96 96" className="w-16 h-16" aria-hidden>
      <rect x="24" y="12" width="44" height="60" rx="4" fill="white" stroke="#0f1e3d" strokeWidth="2" />
      <line x1="32" y1="28" x2="60" y2="28" stroke="#0f1e3d" strokeWidth="2" strokeLinecap="round" opacity="0.4" />
      <line x1="32" y1="38" x2="60" y2="38" stroke="#0f1e3d" strokeWidth="2" strokeLinecap="round" opacity="0.4" />
      <line x1="32" y1="48" x2="48" y2="48" stroke="#0f1e3d" strokeWidth="2" strokeLinecap="round" opacity="0.4" />
      <circle cx="66" cy="66" r="18" fill="#e7edf3" stroke="#0f1e3d" strokeWidth="2" />
      <path d="M58 66l6 6 12-12" fill="none" stroke="#0f1e3d" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// 여러 문서를 겹겹이 감사한다는 의미의 라인아트 일러스트 (B2B)
function AuditStackIllustration() {
  return (
    <svg viewBox="0 0 96 96" className="w-16 h-16" aria-hidden>
      <rect x="14" y="24" width="44" height="56" rx="4" fill="none" stroke="white" strokeOpacity="0.35" strokeWidth="2" />
      <rect x="22" y="16" width="44" height="56" rx="4" fill="none" stroke="white" strokeOpacity="0.6" strokeWidth="2" />
      <rect x="30" y="8" width="44" height="56" rx="4" fill="white" />
      <line x1="38" y1="22" x2="66" y2="22" stroke="#0f1e3d" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
      <line x1="38" y1="32" x2="66" y2="32" stroke="#0f1e3d" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
      <line x1="38" y1="42" x2="56" y2="42" stroke="#0f1e3d" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
      <circle cx="66" cy="50" r="3" fill="#dc2626" />
    </svg>
  );
}

export function AudienceSection() {
  return (
    <section
      id="audience"
      aria-labelledby="audience-heading"
      className="py-24 px-4 bg-white"
    >
      <div className="max-w-5xl mx-auto">
        <h2
          id="audience-heading"
          className="text-2xl md:text-3xl font-bold text-slate-900 text-center mb-10"
        >
          누구를 위한 서비스인가요?
        </h2>

        <div className="grid md:grid-cols-2 gap-5">
          {/* B2C */}
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-8 space-y-4">
            <QuickCheckIllustration />
            <span className="inline-block text-xs font-semibold bg-navy text-white rounded-full px-3 py-1">
              개인 소비자 (B2C)
            </span>
            <h3 className="text-2xl font-bold text-slate-900 leading-snug">
              약관 한 장,
              <br />
              핵심만 빠르게
            </h3>
            <p className="text-slate-500 text-sm leading-relaxed">
              가입 전 약관, 서비스 이용약관, 구독 해지 조항 — 불리한 조항을 요약 판정 카드 한 장으로 확인
            </p>
            <Link
              href="/analyze"
              className="inline-flex items-center gap-2 mt-2 px-5 py-2.5 bg-navy hover:opacity-90 text-white rounded-xl text-sm font-semibold transition-opacity"
            >
              약관 붙여넣기로 시작 <span aria-hidden>→</span>
            </Link>
          </div>

          {/* B2B */}
          <div className="bg-navy rounded-2xl p-8 space-y-4">
            <AuditStackIllustration />
            <span className="inline-block text-xs font-semibold bg-white/10 text-white rounded-full px-3 py-1">
              기업 법무팀 (B2B)
            </span>
            <h3 className="text-2xl font-bold text-white leading-snug">
              약관 전체,
              <br />
              감사 수준 검토
            </h3>
            <p className="text-navy-soft/80 text-sm leading-relaxed">
              자사 약관 사전 검토, 경쟁사 약관 비교 분석 — 사이드바 + 법령 인용 + PDF 리포트 전체 플로우
            </p>
            <Link
              href="/login"
              className="inline-flex items-center gap-2 mt-2 px-5 py-2.5 bg-white hover:bg-slate-100 text-navy rounded-xl text-sm font-semibold transition-colors"
            >
              도입 문의하기 <span aria-hidden>→</span>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
