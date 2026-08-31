// frontend/lib/auth.ts
import type { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import PostgresAdapter from "@auth/pg-adapter";
import { db } from "@/lib/db";

// backend가 psycopg2로 쓰는 것과 같은 Postgres를, lib/db.ts의 공유 Pool로 붙는다
// (예전엔 여기서 별도 Pool을 새로 만들었는데, dev 핫리로드 가드가 없어서 로그인
// 관련 파일을 고칠 때마다 커넥션이 계속 쌓였다 — lib/db.ts 참고). 어댑터가
// 요구하는 users/accounts/sessions/verification_token 스키마는
// backend/db/migrations의 alembic 마이그레이션으로 관리한다.

export const authOptions: NextAuthOptions = {
  adapter: PostgresAdapter(db),
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
  callbacks: {
    // 기본 session 콜백은 user.id를 클라이언트 세션에 안 실어준다 — analyses 테이블의
    // user_id 외래키로 쓸 숫자 id가 필요해서 직접 붙인다. next-auth 코어 타입은
    // AdapterUser.id를 항상 string으로 가정하지만(대부분 어댑터가 uuid/cuid를 씀),
    // @auth/pg-adapter는 Postgres SERIAL을 그대로 돌려주므로 런타임엔 number다 —
    // Number()로 명시 변환해 타입과 실제 값을 일치시킨다.
    session({ session, user }) {
      if (session.user) session.user.id = Number(user.id);
      return session;
    },
  },
};
