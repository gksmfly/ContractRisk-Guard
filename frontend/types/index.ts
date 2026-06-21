// frontend/types/index.ts
export type Domain = "해지_조항" | "책임제한_조항" | "해당없음";
export type RiskLevel = "High" | "Medium" | "Low";

export interface EvidenceSpan {
  text: string;
  start: number;
  end: number;
}

export interface AnalyzeResult {
  domain: Domain;
  risk_level: RiskLevel;
  confidence: number;
  evidence_spans: EvidenceSpan[];
  legal_basis: { law: string; article: string; description: string }[];
  reasoning: string;
  fb_check: {
    forward_label: RiskLevel;
    backward_label: RiskLevel;
    is_clean: boolean;
  };
}
