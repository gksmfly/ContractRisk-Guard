/** @jest-environment node */
import { getServerSession } from "next-auth";
import { requireSession } from "@/lib/require-session";

jest.mock("next-auth", () => ({ getServerSession: jest.fn() }));
jest.mock("@/lib/auth", () => ({ authOptions: {} }));

const ENV = { ...process.env };
const mocked = getServerSession as jest.Mock;
const setEnv = (k: string, v: string) => ((process.env as Record<string, string>)[k] = v);
afterEach(() => {
  process.env = { ...ENV };
  mocked.mockReset();
});

describe("requireSession", () => {
  it("개발·로그인 불필요 모드에서는 통과(세션 조회도 안 함)", async () => {
    setEnv("NODE_ENV", "development");
    expect(await requireSession()).toBeNull();
    expect(mocked).not.toHaveBeenCalled();
  });
  it("배포에서 세션이 없으면 401", async () => {
    setEnv("NODE_ENV", "production");
    mocked.mockResolvedValue(null);
    const res = await requireSession();
    expect(res?.status).toBe(401);
  });
  it("배포에서 허용되지 않은 계정은 403", async () => {
    setEnv("NODE_ENV", "production");
    setEnv("ALLOWED_EMAILS", "ok@x.com");
    mocked.mockResolvedValue({ user: { email: "no@x.com" } });
    expect((await requireSession())?.status).toBe(403);
  });
  it("배포에서 허용된 계정은 통과", async () => {
    setEnv("NODE_ENV", "production");
    setEnv("ALLOWED_EMAILS", "ok@x.com");
    mocked.mockResolvedValue({ user: { email: "OK@x.com" } });
    expect(await requireSession()).toBeNull();
  });
});
