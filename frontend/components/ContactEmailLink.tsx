// frontend/components/ContactEmailLink.tsx
"use client";

const CONTACT_EMAIL = "contact@verilex.example";

// mailto: 링크는 기본 메일 앱이 설정 안 된 환경(브라우저 미리보기, 회사 PC 등)에서
// 클릭해도 아무 반응이 없다 — 눈에 보이는 주소 + 복사 버튼을 항상 같이 둬서
// mailto가 안 열려도 사용자가 직접 복사해 보낼 수 있게 한다.
export function ContactEmailLink({ className = "" }: { className?: string }) {
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(CONTACT_EMAIL);
      alert(`이메일 주소를 복사했습니다: ${CONTACT_EMAIL}`);
    } catch {
      // 클립보드 권한이 없으면 그냥 무시 — 옆에 보이는 주소를 직접 긁어 복사하면 된다
    }
  };

  return (
    <span className={className}>
      <a href={`mailto:${CONTACT_EMAIL}`} className="text-navy font-medium hover:underline">
        {CONTACT_EMAIL}
      </a>{" "}
      <button onClick={handleCopy} className="text-slate-400 hover:text-navy underline">
        (복사)
      </button>
    </span>
  );
}
