// frontend/types/next-auth.d.ts
import type { DefaultSession } from "next-auth";

// 히스토리 저장(analyses.user_id)에 필요한 숫자 user id를 세션에서 쓸 수 있게 타입을 확장한다.
// 실제 값 채우는 곳은 lib/auth.ts의 session 콜백. users.id가 Postgres SERIAL(정수)라
// 런타임 값도 number다 — 예전엔 string으로 잘못 선언돼 있었다(값 자체는 문제없이
// 쿼리 파라미터로 들어갔지만, 타입만 실제와 달랐다).
declare module "next-auth" {
  interface Session {
    user: {
      id: number;
    } & DefaultSession["user"];
  }
}
