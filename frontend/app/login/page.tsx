// frontend/app/login/page.tsx
"use client";

import Link from "next/link";
import { signIn } from "next-auth/react";
import { CheckCircle2 } from "lucide-react";
import { Logo } from "@/components/Logo";

const CHECKLIST = [
  "약관규제법 제9조·10조 직접 인용",
  "공정거래위원회 시정조치 판례 대조",
  "고위험 조항 즉시 식별 및 수정 제안",
];

function GoogleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0" aria-hidden>
      <path fill="#4285F4" d="M23.52 12.27c0-.85-.08-1.67-.22-2.45H12v4.64h6.47a5.53 5.53 0 0 1-2.4 3.63v3h3.88c2.27-2.09 3.57-5.17 3.57-8.82Z" />
      <path fill="#34A853" d="M12 24c3.24 0 5.96-1.07 7.95-2.9l-3.88-3.02c-1.08.72-2.45 1.15-4.07 1.15-3.13 0-5.78-2.11-6.73-4.96H1.26v3.11A12 12 0 0 0 12 24Z" />
      <path fill="#FBBC05" d="M5.27 14.27a7.2 7.2 0 0 1 0-4.54v-3.1H1.26a12 12 0 0 0 0 10.75l4.01-3.11Z" />
      <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.44-3.44C17.95 1.19 15.24 0 12 0 7.31 0 3.26 2.69 1.26 6.63l4.01 3.1C6.22 6.86 8.87 4.75 12 4.75Z" />
    </svg>
  );
}

export default function LoginPage() {
  const handleGoogleLogin = () => {
    signIn("google", { callbackUrl: "/dashboard" });
  };

  return (
    <div className="min-h-screen grid md:grid-cols-2">
      {/* Left panel */}
      <div className="bg-navy text-white flex flex-col justify-center px-10 py-16 md:px-16">
        <Link href="/" className="flex items-center gap-2.5 mb-10 w-fit group">
          <div className="p-1.5 bg-white/10 rounded-lg border border-white/20 group-hover:bg-white/20 transition-colors">
            <Logo className="h-4 w-4 text-white" />
          </div>
          <span className="font-bold text-sm">Verilex</span>
        </Link>

        <h1 className="text-3xl md:text-4xl font-extrabold leading-tight mb-4">
          개인도, 기업 법무팀도
          <br />
          하나의 계정으로
        </h1>
        <p className="text-navy-soft/80 text-sm mb-8">
          법령 근거 기반 판단 · 6단계 AI 검증 · PDF 감사 리포트
        </p>

        <ul className="space-y-3">
          {CHECKLIST.map((item) => (
            <li key={item} className="flex items-center gap-2.5 text-sm">
              <span className="w-5 h-5 rounded-full bg-forest flex items-center justify-center shrink-0">
                <CheckCircle2 className="h-3 w-3" aria-hidden />
              </span>
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* Right panel */}
      <div className="flex flex-col justify-center px-8 py-16 md:px-16 bg-white">
        <div className="max-w-sm w-full mx-auto space-y-6">
          <div>
            <h2 className="text-xl font-bold text-slate-900">로그인</h2>
            <p className="text-sm text-slate-500 mt-1.5">
              개인 소비자와 기업 법무팀 모두 Google 계정으로 바로 시작할 수 있습니다.
            </p>
          </div>

          <button
            onClick={handleGoogleLogin}
            className="w-full flex items-center justify-center gap-3 border border-slate-300 hover:bg-slate-50 rounded-xl py-3 text-sm font-medium text-slate-700 transition-colors"
          >
            <GoogleIcon />
            Google로 계속하기
          </button>

          <p className="text-xs text-slate-500 text-center pt-2 border-t border-rule">
            기업 도입 관련 문의는{" "}
            <a href="mailto:contact@verilex.example" className="text-navy font-medium hover:underline">
              여기로 연락해 주세요 →
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
