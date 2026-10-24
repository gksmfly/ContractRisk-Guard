// frontend/lib/access.ts
// 허용 이메일 목록(ALLOWED_EMAILS, 쉼표 구분)에 있는 계정만 로그인시킨다.
// 목록이 비어 있으면 배포(production)에서는 **전원 거부**한다 — 설정을 빠뜨렸을 때 아무 구글 계정이나 들어오는 쪽이 더 위험하다.
// 개발(dev)에서는 목록이 없어도 통과시켜 로컬 개발 흐름을 유지한다.
export function allowedEmails(): string[] {
  return (process.env.ALLOWED_EMAILS ?? "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
}

export function isEmailAllowed(email?: string | null): boolean {
  const list = allowedEmails();
  if (list.length === 0) return process.env.NODE_ENV !== "production";
  return !!email && list.includes(email.trim().toLowerCase());
}

// 분석 API가 로그인을 요구해야 하는지: 배포이거나 REQUIRE_LOGIN=1일 때. 개발 모드의 기존 동작(로그인 없이 호출)은 그대로 둔다.
export function loginRequired(): boolean {
  return process.env.NODE_ENV === "production" || process.env.REQUIRE_LOGIN === "1";
}
