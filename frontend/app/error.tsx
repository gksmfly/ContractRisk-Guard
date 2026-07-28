// frontend/app/error.tsx
"use client";

import { useEffect } from "react";
import Link from "next/link";
import { AlertTriangle, RotateCcw, Shield } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-6 bg-white px-4 text-center">
      <div className="flex items-center gap-2">
        <div className="p-1.5 bg-blue-50 rounded-lg border border-blue-200">
          <Shield className="h-4 w-4 text-blue-600" aria-hidden />
        </div>
        <span className="font-bold text-slate-900 text-sm">
          Contract<span className="text-blue-600">Risk</span> Guard
        </span>
      </div>

      <div className="p-3 bg-red-50 rounded-full">
        <AlertTriangle className="h-6 w-6 text-red-500" aria-hidden />
      </div>

      <div className="space-y-1.5">
        <h1 className="text-lg font-semibold text-slate-900">문제가 발생했습니다</h1>
        <p className="text-sm text-slate-500 max-w-sm">
          페이지를 처리하는 중 예상치 못한 오류가 발생했습니다. 다시 시도하거나 홈으로 돌아가 주세요.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={reset}
          className="flex items-center gap-1.5 text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-4 py-2 transition-colors"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden />
          다시 시도
        </button>
        <Link
          href="/"
          className="text-sm font-medium text-slate-600 hover:text-slate-800 border border-slate-300 hover:border-slate-400 rounded-lg px-4 py-2 transition-colors"
        >
          홈으로
        </Link>
      </div>
    </div>
  );
}
