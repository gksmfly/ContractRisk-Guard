"use client";

import Link from "next/link";
import { useSession } from "next-auth/react";

// 로그인 여부에 따라 문구가 달라지는 "분석 시작" CTA.
// 이미 로그인한 사용자에게 "로그인하고 시작하세요"라고 말하지 않기 위해 분리했다
// (랜딩 히어로·이용방법·최종 CTA 세 군데가 같은 문구를 쓰므로 한 곳에 모음).
export function StartAnalysisCta({
  className,
  withNote = false,
  arrow = false,
}: {
  className?: string;
  withNote?: boolean;
  arrow?: boolean;
}) {
  const { data: session } = useSession();
  const label = session ? "약관 분석 시작하기" : "Google 로그인하고 약관 분석 시작하기";

  return (
    <>
      <Link href="/analyze" className={className}>
        {label}
        {arrow && <span aria-hidden>→</span>}
      </Link>
      {withNote && !session && (
        <p className="text-slate-400 text-xs mt-3">Google 계정 로그인 후 이용할 수 있어요</p>
      )}
    </>
  );
}
