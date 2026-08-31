// frontend/lib/db.ts
import { Pool } from "pg";

// Next.js dev 모드는 파일이 바뀔 때마다 모듈을 다시 평가하므로, 매번 새 Pool을
// 만들면 연결이 계속 쌓인다 — globalThis에 캐싱해서 핫리로드 사이에도 하나만 쓴다.
// lib/auth.ts의 NextAuth adapter도 이 풀을 그대로 재사용한다(전엔 따로 Pool을
// 만들어서 이 가드가 없었다 — 로그인 관련 파일을 고칠 때마다 핫리로드로 커넥션이
// 새로 쌓이는 문제가 있었음).
const globalForDb = globalThis as unknown as { pgPool?: Pool };

export const db =
  globalForDb.pgPool ?? new Pool({ connectionString: process.env.DATABASE_URL });

if (process.env.NODE_ENV !== "production") {
  globalForDb.pgPool = db;
}
