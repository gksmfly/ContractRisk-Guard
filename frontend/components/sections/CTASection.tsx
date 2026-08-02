// frontend/components/sections/CTASection.tsx
import Link from "next/link";
import { ArrowRight, Shield } from "lucide-react";

export function CTASection() {
  return (
    <section
      aria-labelledby="cta-heading"
      className="py-24 px-4 bg-white"
    >
      <div className="max-w-3xl mx-auto text-center space-y-8">
        <div className="relative">
          <div className="absolute inset-0 bg-navy-soft rounded-3xl blur-3xl" />
          <div className="relative bg-gradient-to-br from-navy-soft to-white border border-navy/20 rounded-3xl px-8 py-14 space-y-6 shadow-sm">
            <div className="flex justify-center">
              <div className="p-3 bg-navy-soft border border-navy/20 rounded-2xl">
                <Shield className="h-7 w-7 text-navy" aria-hidden />
              </div>
            </div>

            <div className="space-y-3">
              <h2
                id="cta-heading"
                className="text-3xl md:text-4xl font-extrabold text-slate-900"
              >
                지금 계약서를 검토해보세요
              </h2>
              <p className="text-slate-600 text-lg max-w-lg mx-auto leading-relaxed">
                회원가입 없이, 무료로, 10초 안에.
                <br />
                계약 체결 전 한 번만 확인해도 다릅니다.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center gap-2 px-8 py-3.5 bg-navy hover:opacity-90 text-white rounded-xl font-semibold transition-colors text-sm shadow-lg shadow-navy/20"
              >
                무료로 분석 시작하기
                <ArrowRight className="h-4 w-4" aria-hidden />
              </Link>
              <a
                href="#examples"
                className="inline-flex items-center justify-center gap-2 px-8 py-3.5 bg-white hover:bg-slate-50 text-slate-700 rounded-xl font-medium transition-colors text-sm border border-slate-300"
              >
                분석 사례 보기
              </a>
            </div>

            <p className="text-xs text-slate-400">
              입력한 텍스트는 저장되지 않으며 분석 후 즉시 폐기됩니다.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
