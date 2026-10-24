/** @jest-environment node */
import { allowedEmails, isEmailAllowed, loginRequired } from "@/lib/access";

const ENV = { ...process.env };
afterEach(() => {
  process.env = { ...ENV };
});
const setEnv = (k: string, v?: string) => {
  const e = process.env as Record<string, string | undefined>;
  if (v === undefined) delete e[k];
  else e[k] = v;
};

describe("isEmailAllowed", () => {
  it("목록에 있는 이메일만 통과(대소문자·공백 무시)", () => {
    setEnv("NODE_ENV", "production");
    setEnv("ALLOWED_EMAILS", " A@x.com , b@y.com ");
    expect(allowedEmails()).toEqual(["a@x.com", "b@y.com"]);
    expect(isEmailAllowed("A@X.com")).toBe(true);
    expect(isEmailAllowed("c@z.com")).toBe(false);
    expect(isEmailAllowed(null)).toBe(false);
  });
  it("배포에서 목록이 비어 있으면 전원 거부(안전 쪽 기본값)", () => {
    setEnv("NODE_ENV", "production");
    setEnv("ALLOWED_EMAILS", "");
    expect(isEmailAllowed("a@x.com")).toBe(false);
  });
  it("개발에서 목록이 비어 있으면 통과(기존 동작 유지)", () => {
    setEnv("NODE_ENV", "development");
    setEnv("ALLOWED_EMAILS", undefined);
    expect(isEmailAllowed("a@x.com")).toBe(true);
  });
});

describe("loginRequired", () => {
  it("배포이거나 REQUIRE_LOGIN=1일 때만 요구", () => {
    setEnv("NODE_ENV", "development");
    setEnv("REQUIRE_LOGIN", undefined);
    expect(loginRequired()).toBe(false);
    setEnv("REQUIRE_LOGIN", "1");
    expect(loginRequired()).toBe(true);
    setEnv("REQUIRE_LOGIN", undefined);
    setEnv("NODE_ENV", "production");
    expect(loginRequired()).toBe(true);
  });
});
