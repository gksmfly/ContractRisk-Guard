// frontend/types/index.ts
export type Domain = "해지_조항" | "책임제한_조항" | "해당없음";
export type RiskLevel = "High" | "Medium" | "Low";

export interface EvidenceSpan {
  text: string;
  start: number;
  end: number;
}
