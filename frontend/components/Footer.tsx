// frontend/components/Footer.tsx
import Link from "next/link";
import { Logo } from "@/components/Logo";

export function Footer() {
  return (
    <footer className="border-t border-slate-200 py-10 px-4 bg-white">
      <div className="max-w-5xl mx-auto">
        <div className="flex flex-col md:flex-row items-center justify-between gap-5">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="p-1.5 bg-navy-soft rounded-lg border border-navy/20 group-hover:bg-navy/10 transition-colors">
              <Logo className="h-4 w-4 text-navy" />
            </div>
            <span className="font-bold text-slate-900 text-sm">
              Veri<span className="text-navy">lex</span>
            </span>
          </Link>

          <nav className="flex gap-5 text-xs text-slate-500">
            <a href="#how-to-use" className="hover:text-slate-700 transition-colors">이용 방법</a>
            <a href="#examples" className="hover:text-slate-700 transition-colors">분석 사례</a>
            <a href="#trust" className="hover:text-slate-700 transition-colors">신뢰 근거</a>
            <a href="#faq" className="hover:text-slate-700 transition-colors">자주 묻는 질문</a>
          </nav>

          <p className="text-xs text-slate-400 text-center md:text-right">
            약관규제법 §7·§9 · 민법 §543~553 · §750~766
          </p>
        </div>

        <div className="mt-6 pt-5 border-t border-slate-200 text-center">
          <p className="text-xs text-slate-400 leading-relaxed">
            본 서비스의 분석 결과는 법적 조언을 대체하지 않으며 참고용으로만 활용하시기 바랍니다.
            최종 계약서 검토는 반드시 법률 전문가와 함께 진행하세요.
          </p>
        </div>
      </div>
    </footer>
  );
}
