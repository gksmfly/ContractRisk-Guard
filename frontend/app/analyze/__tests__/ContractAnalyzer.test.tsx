// frontend/app/analyze/__tests__/ContractAnalyzer.test.tsx
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import type { Session } from "next-auth";
import {
  ContractAnalyzer,
  HighlightText,
  SummaryBar,
} from "@/app/analyze/ContractAnalyzer";
import type { FullAnalyzeResult } from "@/app/api/analyze-full/route";

// 실제 SessionProvider는 마운트 시 /api/auth/session으로 fetch를 시도해서 이 파일의
// SSE 스트림 mock(global.fetch)과 충돌한다 — useSession 훅 자체를 모킹해서 피한다.
jest.mock("next-auth/react", () => ({
  useSession: jest.fn(),
}));

function mockSession(session: Session | null) {
  (useSession as jest.Mock).mockReturnValue({
    data: session,
    status: session ? "authenticated" : "unauthenticated",
  });
}

function renderWithClient(ui: React.ReactElement, session: Session | null = null) {
  mockSession(session);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>
  );
}

// /api/analyze-stream이 실제로 흘려보내는 SSE 청크("data: {...}\n\n")를 흉내 낸다.
function encodeSSE(event: unknown) {
  return new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
}

// 이벤트를 전부 이미 큐에 넣고 바로 닫는 스트림 — 중간 진행 상태를 볼 필요 없는 테스트용.
function mockStreamResponse(events: unknown[]) {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) controller.enqueue(encodeSSE(event));
      controller.close();
    },
  });
  return { ok: true, body: stream };
}

// push()/close()로 외부에서 타이밍을 제어할 수 있는 스트림 — "분석 중" 상태를 검증할 때 쓴다.
function createControllableStream() {
  let controllerRef!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller;
    },
  });
  return {
    stream,
    push: (event: unknown) => controllerRef.enqueue(encodeSSE(event)),
    close: () => controllerRef.close(),
  };
}

describe("SummaryBar", () => {
  it("total_clauses가 0이어도 NaN을 렌더링하지 않는다", () => {
    const result: FullAnalyzeResult = {
      total_clauses: 0,
      review_count: 0,
      high_count: 0,
      medium_count: 0,
      low_count: 0,
      clauses: [],
    };
    render(<SummaryBar result={result} />);

    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
    expect(
      screen.getByText("계약서에서 분석 가능한 조항을 찾지 못했습니다. 계약서 전문을 다시 확인해 주세요.")
    ).toBeInTheDocument();
  });

  // 위험도 3단계(고/중/저 %)를 검증하던 테스트를 대체한다 — 2026-08-31에 등급 자체가
  // 사라졌다(조 multi-label 모델에 risk 헤드가 없다). 지금 화면이 주장하는 것은
  // "입력 N건 중 M건을 확인해야 한다"이고, **나머지는 '안전'이 아니라 '확인되지 않음'** 이다.
  it("확인 필요 비율의 분모는 입력 조항 수이고, 나머지는 '확인되지 않음'으로 센다", () => {
    const result: FullAnalyzeResult = {
      total_clauses: 1,
      review_count: 1,
      input_clauses: 4,   // 입력 4건 중 1건만 확인 필요 → 25%
      clauses: [],
    };
    render(<SummaryBar result={result} />);

    expect(screen.getByText("확인 필요 1건 / 입력 4건 (25%)")).toBeInTheDocument();

    // 확인 필요 1 / 확인되지 않음 3 — 두 칸이 각자 제 값을 든다.
    const review = screen.getByText("확인 필요").parentElement!;
    expect(within(review).getByText("1")).toBeInTheDocument();
    const unchecked = screen.getByText("확인되지 않음").parentElement!;
    expect(within(unchecked).getByText("3")).toBeInTheDocument();

    // "안전"·"저위험"으로 읽힐 문구가 화면에 없어야 한다 — 거짓 안심을 막는 것이 이 설계의 요점이다.
    expect(screen.queryByText(/안전|저위험|고위험|중위험/)).not.toBeInTheDocument();
  });
});

describe("HighlightText", () => {
  it("evidence span이 없으면 원문을 그대로 렌더링한다", () => {
    render(<HighlightText text="사전 통지 없이 해지" spans={[]} />);
    expect(screen.getByText("사전 통지 없이 해지")).toBeInTheDocument();
  });

  it("evidence span 구간을 <mark>로 강조한다", () => {
    const text = "사전 통지 없이 해지할 수 있다";
    const spans = [{ text: "사전 통지 없이", start: 0, end: 8 }];
    const { container } = render(<HighlightText text={text} spans={spans} />);

    const mark = container.querySelector("mark");
    expect(mark).toHaveTextContent("사전 통지 없이");
  });
});

describe("ContractAnalyzer 전체 흐름", () => {
  const mockResult: FullAnalyzeResult = {
    total_clauses: 2,
    review_count: 2,
    high_count: 1,
    medium_count: 0,
    low_count: 1,
    clauses: [
      {
        id: 1,
        original: "회사는 사전 통지 없이 계약을 해지할 수 있다.",
        domain: "해지_조항",
        articles: ["제9조"],
      needs_review: true,
      risk_level: "High",
        confidence_band: "높음",
        confidence_band_accuracy: 0.575,
        evidence_spans: [],
        legal_basis: [{ law: "약관규제법", article: "제9조", description: "부당 해지권 무효" }],
        reasoning: "사전 통지 없는 해지권 부여",
        verified: true,
        redteam_note: "유사 사례 대비 편향 없음",
        evidence_verified: true,
      },
      {
        id: 2,
        original: "당사자는 30일 전 서면 통지로 계약을 해지할 수 있다.",
        domain: "해지_조항",
        articles: ["제9조"],
      needs_review: true,
      risk_level: "Low",
        confidence_band: "중간",
        confidence_band_accuracy: 0.461,
        evidence_spans: [],
        legal_basis: [],
        reasoning: "",
        verified: false,
        redteam_note: "",
        evidence_verified: false,
      },
    ],
  };

  beforeEach(() => {
    global.fetch = jest.fn().mockResolvedValue(
      mockStreamResponse([
        { type: "progress", index: 1, total: 2, articles: ["제9조"],
      needs_review: true,
      risk_level: "High", domain: "해지_조항" },
        { type: "progress", index: 2, total: 2, articles: ["제9조"],
      needs_review: true,
      risk_level: "Low", domain: "해지_조항" },
        { type: "done", result: mockResult },
      ])
    ) as unknown as typeof fetch;
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it("분석하면 기본은 사이드바 없이 카드가 쌓이고, '자세히' 토글로 사이드바+상세 패널로 전환된다", async () => {
    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "회사는 사전 통지 없이 계약을 해지할 수 있는 조항을 포함한다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));

    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    expect(global.fetch).toHaveBeenCalledWith(
      "/api/analyze-stream",
      expect.objectContaining({ method: "POST" })
    );

    // 기본값(간단히 보기)은 사이드바 없이 카드가 쌓인다.
    let workspace = within(screen.getByTestId("clause-workspace"));
    expect(workspace.getByText(/회사는 사전 통지 없이/)).toBeInTheDocument();
    expect(workspace.getByText(/30일 전 서면 통지/)).toBeInTheDocument();
    expect(screen.queryByRole("navigation", { name: "분석된 조항 목록" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tablist", { name: "위험도 필터" })).not.toBeInTheDocument();

    // "자세히" 토글을 누르면 사이드바+상세 패널로 전환된다.
    await user.click(screen.getByRole("tab", { name: "자세히" }));

    workspace = within(screen.getByTestId("clause-workspace"));
    const sidebar = within(workspace.getByRole("navigation", { name: "분석된 조항 목록" }));

    // 기본 선택은 첫 조항 — 상세 패널에 "두 판단 일치" 배지가 보인다.
    expect(workspace.getByText(/회사는 사전 통지 없이/)).toBeInTheDocument();
    expect(workspace.getByText("두 판단 일치")).toBeInTheDocument();
    expect(workspace.queryByText("근거 미확정")).not.toBeInTheDocument();
    // 헤더가 **모델이 지목한 조**를 보여준다. `약관규제법 제9조`는 관련 조문 인용
    // (LegalQuote의 <span>)에도 나오므로 태그로 갈라야 한다 — 그냥 찾으면 2건이 걸린다.
    expect(
      workspace.getByText(
        (_, el) => el?.tagName === "P" && el.textContent === "약관규제법 제9조"
      )
    ).toBeInTheDocument();

    // 사이드바에서 두 번째 조항(근거 미확정)을 선택하면 상세가 전환된다.
    await user.click(sidebar.getByRole("button", { name: /§2/ }));

    expect(workspace.getByText(/30일 전 서면 통지/)).toBeInTheDocument();
    expect(workspace.getByText("근거 미확정")).toBeInTheDocument();
    expect(workspace.queryByText(/회사는 사전 통지 없이/)).not.toBeInTheDocument();

    // 위험도 필터를 검증하던 블록을 제거했다 — 2026-08-31에 등급을 내지 않기로 하면서
    // 필터할 축 자체가 사라졌다(ContractAnalyzer.tsx의 "위험도 필터 제거" 주석 참고).
    // 등급이 화면에 되살아나면 이 자리에서 다시 걸리도록 아래를 남긴다.
    expect(screen.queryByRole("tablist", { name: "위험도 필터" })).not.toBeInTheDocument();
  });

  it("미니맵을 클릭하면 해당 조항 상세로 전환된다", async () => {
    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "회사는 사전 통지 없이 계약을 해지할 수 있는 조항을 포함한다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    // 위험도 미니맵은 "자세히" 보기의 상세 패널 안에서만 렌더링된다.
    await user.click(screen.getByRole("tab", { name: "자세히" }));

    const minimap = within(screen.getByRole("group", { name: "조항별 위험도 미니맵" }));
    await user.click(minimap.getByRole("button", { name: /§2/ }));

    const workspace = within(screen.getByTestId("clause-workspace"));
    expect(workspace.getByText(/30일 전 서면 통지/)).toBeInTheDocument();
  });

  it("법령 인용의 '원문에서 보기'를 누르면 원문으로 스크롤하고 강조 표시한다", async () => {
    const localMock: FullAnalyzeResult = {
      total_clauses: 1,
      review_count: 1,
      high_count: 1,
      medium_count: 0,
      low_count: 0,
      clauses: [
        {
          id: 1,
          original: "회사는 사전 통지 없이 계약을 해지할 수 있다.",
          domain: "해지_조항",
          articles: ["제9조"],
      needs_review: true,
      risk_level: "High",
          confidence_band: "높음",
          confidence_band_accuracy: 0.575,
          evidence_spans: [{ text: "사전 통지 없이", start: 4, end: 12 }],
          legal_basis: [{ law: "약관규제법", article: "제9조", description: "부당 해지권 무효" }],
          reasoning: "",
          verified: false,
          redteam_note: "",
          evidence_verified: true,
        },
      ],
    };
    global.fetch = jest.fn().mockResolvedValue(
      mockStreamResponse([
        { type: "progress", index: 1, total: 1, articles: ["제9조"],
      needs_review: true,
      risk_level: "High", domain: "해지_조항" },
        { type: "done", result: localMock },
      ])
    ) as unknown as typeof fetch;
    Element.prototype.scrollIntoView = jest.fn();

    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "아무 계약서 텍스트나 20자 이상 입력하는 테스트용 문장입니다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    const workspace = within(screen.getByTestId("clause-workspace"));
    await user.click(workspace.getByRole("button", { name: "원문에서 보기" }));

    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
    const mark = workspace.getAllByText("사전 통지 없이")[0];
    expect(mark.tagName.toLowerCase()).toBe("mark");
    expect(mark).toHaveClass("bg-seal-soft");
  });

  it("분석 중에는 조항 단위 실시간 진행 상태가 표시된다", async () => {
    const { stream, push, close } = createControllableStream();
    global.fetch = jest.fn().mockResolvedValue({ ok: true, body: stream }) as unknown as typeof fetch;

    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "계약서 조항 예시 텍스트를 충분히 길게 입력합니다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));

    expect(await screen.findByText("약관을 분석하고 있습니다")).toBeInTheDocument();
    // 첫 진행 이벤트가 오기 전에는 조항 수를 모르는 불확정 상태를 보여준다 — 가짜로 지어내지 않는다.
    expect(screen.getByText("조항을 분리하고 있습니다...")).toBeInTheDocument();

    push({ type: "progress", index: 1, total: 2, articles: ["제9조"],
      needs_review: true,
      risk_level: "High", domain: "해지_조항" });
    await waitFor(() => expect(screen.getByText("조항 1")).toBeInTheDocument());

    push({ type: "progress", index: 2, total: 2, articles: ["제9조"],
      needs_review: true,
      risk_level: "Low", domain: "해지_조항" });
    push({ type: "done", result: mockResult });
    close();
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());
    expect(screen.queryByText("조항 1")).not.toBeInTheDocument();
  });

  it("PDF 리포트 버튼은 항상 노출되고 누르면 인쇄를 호출한다", async () => {
    const printSpy = jest.spyOn(window, "print").mockImplementation(() => {});
    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "계약서 조항 예시 텍스트를 충분히 길게 입력합니다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    // 간단히 보기(기본값)에서도 PDF 리포트 버튼은 노출된다 — 모드로 가려지지 않는다.
    await user.click(screen.getByRole("button", { name: /PDF 리포트/ }));

    expect(printSpy).toHaveBeenCalledTimes(1);
    printSpy.mockRestore();
  });

  it("로그인하지 않은 사용자에게는 결과 하단에 저장 유도 배너가 보인다", async () => {
    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />, null);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "계약서 조항 예시 텍스트를 충분히 길게 입력합니다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    expect(screen.getByText("이 분석 결과를 저장할까요?")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "로그인하고 저장하기" })).toHaveAttribute("href", "/login");
  });

  it("로그인한 사용자는 '저장하기'를 눌러 결과를 히스토리에 저장할 수 있다", async () => {
    const user = userEvent.setup();
    const session: Session = { user: { id: 1, name: "테스트" }, expires: "2099-01-01" };

    global.fetch = jest.fn().mockImplementation((url: string) => {
      if (url === "/api/history") {
        return Promise.resolve({ ok: true, json: async () => ({ id: 1, created_at: "2026-08-16" }) });
      }
      return Promise.resolve(
        mockStreamResponse([
          { type: "progress", index: 1, total: 2, articles: ["제9조"],
      needs_review: true,
      risk_level: "High", domain: "해지_조항" },
          { type: "progress", index: 2, total: 2, articles: ["제9조"],
      needs_review: true,
      risk_level: "Low", domain: "해지_조항" },
          { type: "done", result: mockResult },
        ])
      );
    }) as unknown as typeof fetch;

    renderWithClient(<ContractAnalyzer />, session);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "계약서 조항 예시 텍스트를 충분히 길게 입력합니다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    // 로그인하지 않았을 때 보이던 "로그인하고 저장하기" 링크 대신 바로 저장 버튼이 보인다.
    expect(screen.queryByRole("link", { name: "로그인하고 저장하기" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "저장하기" }));

    await waitFor(() => expect(screen.getByText("히스토리에 저장했습니다")).toBeInTheDocument());
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/history",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("PDF 업로드 후 분석하면 결과가 표시되고 스트리밍 엔드포인트를 호출한다", async () => {
    global.fetch = jest.fn().mockResolvedValue(
      mockStreamResponse([
        { type: "progress", index: 1, total: 2, articles: ["제9조"],
      needs_review: true,
      risk_level: "High", domain: "해지_조항" },
        { type: "progress", index: 2, total: 2, articles: ["제9조"],
      needs_review: true,
      risk_level: "Low", domain: "해지_조항" },
        { type: "done", result: mockResult },
      ])
    ) as unknown as typeof fetch;

    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    await user.click(screen.getByRole("tab", { name: "PDF 업로드" }));
    const file = new File(["dummy pdf content"], "약관.pdf", { type: "application/pdf" });
    const input = screen
      .getByRole("button", { name: "계약서 파일 업로드 영역" })
      .querySelector('input[type="file"]') as HTMLInputElement;
    await userEvent.upload(input, file);

    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));
    // PDF 업로드 시 결과 헤더에는 "분석 결과" 대신 업로드한 파일명이 표시된다.
    await waitFor(() => expect(screen.getByText("약관.pdf")).toBeInTheDocument());
    expect(screen.getByText("분석 완료")).toBeInTheDocument();

    expect(global.fetch).toHaveBeenCalledWith(
      "/api/analyze-pdf-stream",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("업로드 파일이 최대 크기를 넘으면 분석을 진행하지 않는다", async () => {
    const alertSpy = jest.spyOn(window, "alert").mockImplementation(() => {});
    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    // 드롭존은 "PDF 업로드" 탭 안에 있다.
    await user.click(screen.getByRole("tab", { name: "PDF 업로드" }));

    const bigFile = new File(["x".repeat(10)], "huge.pdf", { type: "application/pdf" });
    Object.defineProperty(bigFile, "size", { value: 11 * 1024 * 1024 });

    const input = screen
      .getByRole("button", { name: "계약서 파일 업로드 영역" })
      .querySelector('input[type="file"]') as HTMLInputElement;
    await userEvent.upload(input, bigFile);

    expect(alertSpy).toHaveBeenCalledWith(
      expect.stringContaining("파일 크기는")
    );
    expect(global.fetch).not.toHaveBeenCalled();

    alertSpy.mockRestore();
  });
});
