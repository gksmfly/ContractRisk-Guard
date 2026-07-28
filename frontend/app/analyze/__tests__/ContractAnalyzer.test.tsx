// frontend/app/analyze/__tests__/ContractAnalyzer.test.tsx
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  ContractAnalyzer,
  HighlightText,
  SummaryBar,
} from "@/app/analyze/ContractAnalyzer";
import type { FullAnalyzeResult } from "@/app/api/analyze-full/route";

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>
  );
}

describe("SummaryBar", () => {
  it("total_clauses가 0이어도 NaN을 렌더링하지 않는다", () => {
    const result: FullAnalyzeResult = {
      total_clauses: 0,
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

  it("정상 케이스에서 위험도 비율을 정확히 계산한다", () => {
    const result: FullAnalyzeResult = {
      total_clauses: 4,
      high_count: 1,
      medium_count: 1,
      low_count: 2,
      clauses: [],
    };
    render(<SummaryBar result={result} />);

    expect(screen.getByText("고위험 25%")).toBeInTheDocument();
    expect(screen.getByText("중위험 25%")).toBeInTheDocument();
    expect(screen.getByText("저위험 50%")).toBeInTheDocument();
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
    high_count: 1,
    medium_count: 0,
    low_count: 1,
    clauses: [
      {
        id: 1,
        original: "회사는 사전 통지 없이 계약을 해지할 수 있다.",
        domain: "해지_조항",
        risk_level: "High",
        confidence: 0.9,
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
        risk_level: "Low",
        confidence: 0.8,
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
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResult,
    }) as unknown as typeof fetch;
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it("텍스트 입력 후 분석하면 결과가 사이드바+상세 패널로 노출된다", async () => {
    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "회사는 사전 통지 없이 계약을 해지할 수 있는 조항을 포함한다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));

    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    expect(global.fetch).toHaveBeenCalledWith(
      "/api/analyze-full",
      expect.objectContaining({ method: "POST" })
    );

    // 인쇄용 전체 목록이 DOM에 함께 존재하므로(화면에서는 CSS로 숨김), 상호작용 영역으로 범위를 좁혀 검증한다.
    const workspace = () => within(screen.getByTestId("clause-workspace"));

    // 기본 선택은 첫 조항(고위험, 검증됨) — 상세 패널에 검증 배지가 보인다.
    expect(workspace().getByText(/회사는 사전 통지 없이/)).toBeInTheDocument();
    expect(workspace().getByText("검증됨")).toBeInTheDocument();
    expect(workspace().queryByText("근거 미확정")).not.toBeInTheDocument();
    expect(workspace().getByText(/약관규제법 제9조/)).toBeInTheDocument();

    // 사이드바에서 두 번째 조항(저위험, 근거 미확정)을 선택하면 상세가 전환된다.
    await user.click(workspace().getByRole("button", { name: /제2항/ }));

    expect(workspace().getByText(/30일 전 서면 통지/)).toBeInTheDocument();
    expect(workspace().getByText("근거 미확정")).toBeInTheDocument();
    expect(workspace().queryByText(/회사는 사전 통지 없이/)).not.toBeInTheDocument();

    // 고위험 필터를 적용하면 저위험 조항은 목록/상세에서 사라진다.
    await user.click(screen.getByRole("tab", { name: /고위험/ }));

    expect(workspace().getByText(/회사는 사전 통지 없이/)).toBeInTheDocument();
    expect(workspace().queryByRole("button", { name: /제2항/ })).not.toBeInTheDocument();
  });

  it("미니맵을 클릭하면 해당 조항 상세로 전환된다", async () => {
    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "회사는 사전 통지 없이 계약을 해지할 수 있는 조항을 포함한다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    const minimap = within(screen.getByRole("group", { name: "조항별 위험도 미니맵" }));
    await user.click(minimap.getByRole("button", { name: /제2항/ }));

    const workspace = within(screen.getByTestId("clause-workspace"));
    expect(workspace.getByText(/30일 전 서면 통지/)).toBeInTheDocument();
  });

  it("법령 인용의 '원문에서 보기'를 누르면 원문으로 스크롤하고 강조 표시한다", async () => {
    const localMock: FullAnalyzeResult = {
      total_clauses: 1,
      high_count: 1,
      medium_count: 0,
      low_count: 0,
      clauses: [
        {
          id: 1,
          original: "회사는 사전 통지 없이 계약을 해지할 수 있다.",
          domain: "해지_조항",
          risk_level: "High",
          confidence: 0.9,
          evidence_spans: [{ text: "사전 통지 없이", start: 4, end: 12 }],
          legal_basis: [{ law: "약관규제법", article: "제9조", description: "부당 해지권 무효" }],
          reasoning: "",
          verified: false,
          redteam_note: "",
          evidence_verified: true,
        },
      ],
    };
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => localMock,
    }) as unknown as typeof fetch;
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

  it("분석 중에는 6-agent 진행 상태가 표시된다", async () => {
    let resolveFetch!: (value: unknown) => void;
    global.fetch = jest.fn().mockImplementation(
      () => new Promise((resolve) => { resolveFetch = resolve; })
    ) as unknown as typeof fetch;

    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "계약서 조항 예시 텍스트를 충분히 길게 입력합니다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));

    expect(await screen.findByText("6-Agent 파이프라인 진행 중")).toBeInTheDocument();
    expect(screen.getByText(/조항 1차 분석/)).toBeInTheDocument();

    resolveFetch({ ok: true, json: async () => mockResult });
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());
    expect(screen.queryByText("6-Agent 파이프라인 진행 중")).not.toBeInTheDocument();
  });

  it("리포트 저장 버튼을 누르면 인쇄를 호출한다", async () => {
    const printSpy = jest.spyOn(window, "print").mockImplementation(() => {});
    const user = userEvent.setup();
    renderWithClient(<ContractAnalyzer />);

    const textarea = screen.getByLabelText("분석할 계약서 전문 입력");
    await user.type(textarea, "계약서 조항 예시 텍스트를 충분히 길게 입력합니다.");
    await user.click(screen.getByRole("button", { name: "계약서 전체 분석 시작" }));
    await waitFor(() => expect(screen.getByText("분석 결과")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /리포트 저장/ }));

    expect(printSpy).toHaveBeenCalledTimes(1);
    printSpy.mockRestore();
  });

  it("업로드 파일이 최대 크기를 넘으면 분석을 진행하지 않는다", async () => {
    const alertSpy = jest.spyOn(window, "alert").mockImplementation(() => {});
    renderWithClient(<ContractAnalyzer />);

    const bigFile = new File(["x".repeat(10)], "huge.txt", { type: "text/plain" });
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
