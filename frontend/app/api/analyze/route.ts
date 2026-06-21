// frontend/app/api/analyze/route.ts
import { NextRequest, NextResponse } from "next/server";
import type { AnalyzeResult, Domain, RiskLevel } from "@/types";

const HIGH_TERMINATION_KEYWORDS = [
  "즉시 해지",
  "일방적",
  "통보 없이",
  "사전 통지 없이",
  "이유 없이",
  "임의로 해지",
];
const LIABILITY_KEYWORDS = [
  "책임을 지지 않",
  "배상 책임 없",
  "면책",
  "귀책사유 불문",
  "어떠한 손해도",
];

function classify(text: string): { domain: Domain; risk: RiskLevel } {
  const isTermination = /해지|해제|계약.*종료|종료.*계약/u.test(text);
  const isLiability = /책임.*제한|면책|배상.*없|손해.*불|책임지지/u.test(text);

  let domain: Domain = "해당없음";
  if (isTermination) domain = "해지_조항";
  else if (isLiability) domain = "책임제한_조항";

  const highHit =
    domain === "해지_조항"
      ? HIGH_TERMINATION_KEYWORDS.some((k) => text.includes(k))
      : LIABILITY_KEYWORDS.some((k) => text.includes(k));

  const risk: RiskLevel = highHit
    ? "High"
    : domain !== "해당없음"
    ? "Medium"
    : "Low";

  return { domain, risk };
}

function findEvidenceSpans(text: string, domain: Domain) {
  const patterns =
    domain === "해지_조항"
      ? [...HIGH_TERMINATION_KEYWORDS, "해지", "해제", "종료"]
      : [...LIABILITY_KEYWORDS, "책임", "면책", "배상"];

  const spans: { text: string; start: number; end: number }[] = [];
  for (const pattern of patterns) {
    let idx = text.indexOf(pattern);
    while (idx !== -1) {
      spans.push({ text: pattern, start: idx, end: idx + pattern.length });
      idx = text.indexOf(pattern, idx + 1);
    }
  }
  return spans.slice(0, 5);
}

const LEGAL_BASIS: Record<Domain, { law: string; article: string; description: string }[]> = {
  해지_조항: [
    {
      law: "약관규제법",
      article: "제9조",
      description: "고객에게 부당하게 불리한 해제·해지권 부여 조항 무효",
    },
    {
      law: "민법",
      article: "제543~553조",
      description: "계약 해제·해지의 일반 요건 및 효과",
    },
  ],
  책임제한_조항: [
    {
      law: "약관규제법",
      article: "제7조",
      description: "사업자 책임 부당 면제·제한 조항 무효",
    },
    {
      law: "민법",
      article: "제750~766조",
      description: "불법행위 손해배상 책임",
    },
  ],
  해당없음: [],
};

const SAMPLE_REASONING: Record<Domain, string> = {
  해지_조항:
    "본 조항은 계약 당사자 일방에게 사전 통지 없이 계약을 해지할 수 있는 권한을 부여합니다. 약관규제법 제9조에 따르면 고객에게 부당하게 불리한 계약 해제·해지권을 사업자에게 부여하는 조항은 무효입니다. 실제 시정조치 사례와 비교 분석 결과 고위험 패턴으로 분류됩니다.",
  책임제한_조항:
    "본 조항은 사업자의 손해배상 책임을 포괄적으로 면제 또는 제한하는 내용을 포함합니다. 약관규제법 제7조는 사업자의 고의·과실로 인한 법률상 책임을 배제하는 조항을 무효로 규정합니다. 판례 분석 결과 소비자 피해 유발 가능성이 높은 조항입니다.",
  해당없음: "분석된 조항에서 계약 해지 또는 책임제한 관련 법적 리스크 패턴이 발견되지 않았습니다.",
};

export async function POST(req: NextRequest) {
  const { text } = await req.json();

  if (!text || text.trim().length < 10) {
    return NextResponse.json({ error: "조항 텍스트가 너무 짧습니다." }, { status: 400 });
  }

  await new Promise((r) => setTimeout(r, 800 + Math.random() * 400));

  const { domain, risk } = classify(text);
  const evidenceSpans = findEvidenceSpans(text, domain);
  const forwardLabel = risk;
  const backwardLabel =
    evidenceSpans.length > 0 ? risk : risk === "High" ? "Medium" : risk;

  const result: AnalyzeResult = {
    domain,
    risk_level: risk,
    confidence: domain === "해당없음" ? 0.62 : risk === "High" ? 0.91 : 0.78,
    evidence_spans: evidenceSpans,
    legal_basis: LEGAL_BASIS[domain],
    reasoning: SAMPLE_REASONING[domain],
    fb_check: {
      forward_label: forwardLabel,
      backward_label: backwardLabel,
      is_clean: forwardLabel === backwardLabel,
    },
  };

  return NextResponse.json(result);
}
