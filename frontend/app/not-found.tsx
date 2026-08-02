// frontend/app/not-found.tsx
import Link from "next/link";
import { FileQuestion } from "lucide-react";
import { Logo } from "@/components/Logo";

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-6 bg-white px-4 text-center">
      <div className="flex items-center gap-2">
        <div className="p-1.5 bg-navy-soft rounded-lg border border-navy/20">
          <Logo className="h-4 w-4 text-navy" />
        </div>
        <span className="font-bold text-slate-900 text-sm">
          Veri<span className="text-navy">lex</span>
        </span>
      </div>

      <div className="p-3 bg-slate-100 rounded-full">
        <FileQuestion className="h-6 w-6 text-slate-500" aria-hidden />
      </div>

      <div className="space-y-1.5">
        <h1 className="text-lg font-semibold text-slate-900">페이지를 찾을 수 없습니다</h1>
        <p className="text-sm text-slate-500 max-w-sm">
          요청하신 페이지가 존재하지 않거나 이동되었습니다.
        </p>
      </div>

      <Link
        href="/"
        className="text-sm font-medium bg-navy hover:opacity-90 text-white rounded-lg px-4 py-2 transition-opacity"
      >
        홈으로 돌아가기
      </Link>
    </div>
  );
}
