// frontend/lib/auth.ts
import type { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import PostgresAdapter from "@auth/pg-adapter";
import { Pool } from "pg";

// backend가 psycopg2로 쓰는 것과 같은 Postgres를 pg 패키지로 붙는다.
// 어댑터가 요구하는 users/accounts/sessions/verification_token 스키마는
// backend/db/migrations의 alembic 마이그레이션으로 관리한다.
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

export const authOptions: NextAuthOptions = {
  adapter: PostgresAdapter(pool),
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID ?? "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET ?? "",
    }),
  ],
  // adapter를 쓰므로 세션을 DB(sessions 테이블)에 저장한다 — JWT 세션이면
  // adapter가 로그인 시 유저만 만들고 세션 자체는 안 쓰게 된다.
  session: { strategy: "database" },
  pages: {
    signIn: "/login",
  },
};
