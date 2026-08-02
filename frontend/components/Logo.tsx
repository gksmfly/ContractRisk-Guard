// frontend/components/Logo.tsx
// 방패(보호) + 체크마크(검증 — Verilex의 "Veri") 워드마크. 이전 버전은 저울을 가는
// 4~5개 선으로 그려서 내비게이션 크기(16~20px)에서는 뭉개져 보였다 — 굵은 stroke
// 2개짜리 체크마크로 바꿔 작은 크기에서도 또렷하게 읽히게 했다.
export function Logo({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" className={className} aria-hidden>
      <path
        d="M16 4L25.5 8V16C25.5 21.8 21.2 26.7 16 28.5C10.8 26.7 6.5 21.8 6.5 16V8L16 4Z"
        fill="currentColor"
      />
      <path
        d="M11 16.5L14.5 20L21.5 12.5"
        stroke="white"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}
