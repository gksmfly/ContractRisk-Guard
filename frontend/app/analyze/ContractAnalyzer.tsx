// frontend/app/analyze/ContractAnalyzer.tsx
"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import type { FullAnalyzeResult } from "@/app/api/analyze-full/route";
import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  HighlighterIcon,
  Loader2,
  Printer,
  Scale,
  Search,
  Shield,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { MAX_UPLOAD_SIZE_BYTES } from "@/lib/config";

// ── Types ──────────────────────────────────────────────
type RiskLevel = "High" | "Medium" | "Low";
type Clause = FullAnalyzeResult["clauses"][0];

const RISK_CFG: Record<
  RiskLevel,
  {
    label: string;
    color: string;
    bg: string;
    ring: string;
    dotCls: string;
    icon: typeof AlertTriangle;
  }
> = {
  High: {
    label: "고위험",
    color: "text-seal",
    bg: "bg-seal-soft",
    ring: "border-seal",
    dotCls: "bg-seal",
    icon: AlertTriangle,
  },
  Medium: {
    label: "중위험",
    color: "text-ochre",
    bg: "bg-ochre-soft",
    ring: "border-ochre",
    dotCls: "bg-ochre",
    icon: AlertTriangle,
  },
  Low: {
    label: "저위험",
    color: "text-forest",
    bg: "bg-forest-soft",
    ring: "border-forest",
    dotCls: "bg-forest",
    icon: CheckCircle2,
  },
};

// ── Highlight helper ───────────────────────────────────
export function HighlightText({
  text,
  spans,
  flash,
}: {
  text: string;
  spans: { text: string; start: number; end: number }[];
  flash?: boolean;
}) {
  if (!spans.length)
    return <span className="text-slate-700 text-sm leading-relaxed">{text}</span>;

  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const parts: { t: string; hi: boolean }[] = [];
  let cursor = 0;
  for (const s of sorted) {
    if (s.start > cursor) parts.push({ t: text.slice(cursor, s.start), hi: false });
    parts.push({ t: text.slice(s.start, s.end), hi: true });
    cursor = s.end;
  }
  if (cursor < text.length) parts.push({ t: text.slice(cursor), hi: false });

  return (
    <span className="text-sm leading-relaxed">
      {parts.map((p, i) =>
        p.hi ? (
          <mark
            key={i}
            className={`text-seal font-semibold underline decoration-wavy decoration-seal/70 underline-offset-4 rounded px-0.5 transition-colors duration-700 ${
              flash ? "bg-seal-soft" : "bg-transparent"
            }`}
          >
            {p.t}
          </mark>
        ) : (
          <span key={i} className="text-slate-700">{p.t}</span>
        )
      )}
    </span>
  );
}

// ── Clause list item (sidebar) ─────────────────────────
function ClauseListItem({
  clause,
  active,
  onClick,
}: {
  clause: Clause;
  active: boolean;
  onClick: () => void;
}) {
  const cfg = RISK_CFG[clause.risk_level];

  return (
    <button
      onClick={onClick}
      aria-current={active}
      className={`w-full text-left px-3 py-2.5 flex items-center gap-2.5 transition-colors border-l-2 shrink-0 min-w-[180px] md:min-w-0 md:w-full
        ${active ? "bg-navy-soft border-navy" : "border-transparent hover:bg-white"}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${cfg.dotCls}`} aria-hidden />
      <span className="text-xs text-slate-400 font-mono shrink-0">제{clause.id}항</span>
      <span className="flex-1 text-xs text-slate-600 truncate">
        {clause.domain === "해당없음" ? "분류 불가" : clause.domain}
      </span>
      {clause.verified && (
        <CheckCircle2 className="h-3 w-3 text-navy shrink-0" aria-label="검증됨" />
      )}
    </button>
  );
}

// ── Risk minimap ────────────────────────────────────────
function RiskMinimap({
  clauses,
  selectedId,
  onSelect,
}: {
  clauses: Clause[];
  selectedId: number | undefined;
  onSelect: (id: number) => void;
}) {
  if (clauses.length <= 1) return null;

  return (
    <div
      role="group"
      aria-label="조항별 위험도 미니맵"
      className="flex items-center gap-1 flex-wrap print:hidden"
    >
      <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wide mr-1 shrink-0">
        미니맵
      </span>
      {clauses.map((c) => {
        const cfg = RISK_CFG[c.risk_level];
        const active = c.id === selectedId;
        return (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            aria-label={`제${c.id}항(${cfg.label})으로 이동`}
            aria-current={active}
            title={`제${c.id}항 · ${cfg.label}`}
            className={`h-2 rounded-sm transition-all ${cfg.dotCls} ${
              active ? "w-6 ring-2 ring-offset-1 ring-navy" : "w-3 opacity-50 hover:opacity-90"
            }`}
          />
        );
      })}
    </div>
  );
}

// ── Legal basis quote block ────────────────────────────
function LegalQuote({
  basis,
  onJumpToEvidence,
}: {
  basis: Clause["legal_basis"][0];
  onJumpToEvidence?: () => void;
}) {
  return (
    <div className="border-y border-rule py-3 print:break-inside-avoid">
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <div className="flex items-center gap-1.5 min-w-0">
          <Scale className="h-3 w-3 text-navy shrink-0" aria-hidden />
          <span className="text-xs font-mono font-semibold text-navy truncate">
            {basis.law} {basis.article}
          </span>
        </div>
        {onJumpToEvidence && (
          <button
            onClick={onJumpToEvidence}
            className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-seal transition-colors shrink-0 print:hidden"
          >
            <HighlighterIcon className="h-3 w-3" aria-hidden />
            원문에서 보기
          </button>
        )}
      </div>
      <blockquote className="italic text-[13px] text-slate-700 leading-relaxed">
        “{basis.description}”
      </blockquote>
    </div>
  );
}

// ── Section label (rule-line heading) ──────────────────
function SectionLabel({ icon: Icon, children }: { icon: typeof Shield; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 text-[11px] font-mono font-semibold uppercase tracking-wide text-navy">
      <Icon className="h-3 w-3 shrink-0" aria-hidden />
      <span className="shrink-0">{children}</span>
      <span className="flex-1 h-px bg-rule" aria-hidden />
    </div>
  );
}

// ── Clause detail panel ─────────────────────────────────
function ClauseDetail({ clause }: { clause: Clause }) {
  const cfg = RISK_CFG[clause.risk_level];
  const [flash, setFlash] = useState(false);
  const originalRef = useRef<HTMLDivElement>(null);

  const handleJumpToEvidence = () => {
    originalRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlash(true);
    window.setTimeout(() => setFlash(false), 1400);
  };

  return (
    <article className="border border-rule bg-white print:break-inside-avoid print:border-slate-300">
      {/* Letterhead bar */}
      <div className="bg-navy text-navy-soft px-5 py-2 flex items-center justify-between gap-3 font-mono text-[10px] tracking-wide">
        <span>AI 리스크 판단 결과</span>
        <span className="flex items-center gap-1.5">
          {clause.verified && (
            <span className="inline-flex items-center gap-1" aria-label="Red-team 검증 통과">
              <CheckCircle2 className="h-3 w-3" aria-hidden /> 검증됨
            </span>
          )}
          {!clause.evidence_verified && <span aria-label="근거 재검색이 필요했던 조항">근거 미확정</span>}
        </span>
      </div>

      {/* Header row */}
      <div className="px-5 py-4 flex items-start justify-between gap-4 border-b border-rule">
        <div className="min-w-0">
          <p className="text-[11px] font-mono text-slate-500">
            제{clause.id}항 · {clause.domain === "해당없음" ? "분류 불가" : clause.domain}
          </p>
          <p className="text-lg font-semibold text-slate-900 mt-0.5">
            위험도 판단: {cfg.label}
          </p>
        </div>
        <div
          className={`shrink-0 w-14 h-14 rounded-full border-2 ${cfg.ring} ${cfg.color} flex items-center justify-center -rotate-6 font-mono`}
          aria-hidden
        >
          <span className="text-[11px] font-bold leading-none text-center">{cfg.label}</span>
        </div>
      </div>

      {/* Original text */}
      <div ref={originalRef} className="px-5 py-4 border-b border-rule scroll-mt-4">
        {clause.evidence_spans.length > 0 ? (
          <HighlightText text={clause.original} spans={clause.evidence_spans} flash={flash} />
        ) : (
          <p className="text-slate-600 text-sm leading-relaxed">{clause.original}</p>
        )}
      </div>

      {/* Detail */}
      <div className="px-5 py-5 space-y-5">
        {/* Legal basis */}
        {clause.legal_basis.length > 0 && (
          <div className="space-y-1">
            <SectionLabel icon={Shield}>적용 법령 원문</SectionLabel>
            <div>
              {clause.legal_basis.map((lb, i) => (
                <LegalQuote
                  key={i}
                  basis={lb}
                  onJumpToEvidence={
                    clause.evidence_spans.length > 0 ? handleJumpToEvidence : undefined
                  }
                />
              ))}
            </div>
          </div>
        )}

        {/* Reasoning */}
        {clause.reasoning && (
          <div className="space-y-2">
            <SectionLabel icon={Zap}>판단 근거</SectionLabel>
            <p className="text-xs text-slate-600 bg-navy-soft border-l-2 border-navy px-3.5 py-2.5 leading-relaxed">
              {clause.reasoning}
            </p>
          </div>
        )}

        {/* Red-team note */}
        {clause.redteam_note && (
          <div className="space-y-2">
            <SectionLabel icon={AlertTriangle}>Red-team 검토 의견</SectionLabel>
            <p className={`text-xs text-slate-600 ${cfg.bg} border-l-2 ${cfg.ring} px-3.5 py-2.5 leading-relaxed`}>
              {clause.redteam_note}
            </p>
          </div>
        )}
      </div>
    </article>
  );
}

// ── Summary bar ───────────────────────────────────────
export function SummaryBar({ result }: { result: FullAnalyzeResult }) {
  const hasClauses = result.total_clauses > 0;
  const highPct = hasClauses ? Math.round((result.high_count / result.total_clauses) * 100) : 0;
  const medPct = hasClauses ? Math.round((result.medium_count / result.total_clauses) * 100) : 0;
  const lowPct = hasClauses ? 100 - highPct - medPct : 0;

  if (!hasClauses) {
    return (
      <div className="bg-white border border-rule p-6 text-center">
        <p className="text-slate-500 text-sm">
          계약서에서 분석 가능한 조항을 찾지 못했습니다. 계약서 전문을 다시 확인해 주세요.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-rule p-6 space-y-4">
      <div className="flex flex-wrap gap-6 justify-between">
        <div>
          <p className="text-slate-500 text-xs mb-1 font-mono uppercase tracking-wide">총 분석 조항</p>
          <p className="text-3xl font-bold text-slate-900">{result.total_clauses}<span className="text-lg text-slate-400 ml-1">건</span></p>
        </div>
        <div className="flex gap-5">
          {[
            { label: "고위험", count: result.high_count, color: "text-seal" },
            { label: "중위험", count: result.medium_count, color: "text-ochre" },
            { label: "저위험", count: result.low_count, color: "text-forest" },
          ].map((s) => (
            <div key={s.label} className="text-center">
              <p className="text-xs text-slate-500 mb-0.5">{s.label}</p>
              <p className={`text-2xl font-bold ${s.color}`}>{s.count}</p>
            </div>
          ))}
        </div>
      </div>
      {/* Risk bar */}
      <div className="space-y-1.5">
        <p className="text-xs text-slate-500 font-mono uppercase tracking-wide">위험도 분포</p>
        <div className="flex h-1.5 overflow-hidden gap-px bg-rule">
          {highPct > 0 && <div className="bg-seal transition-all" style={{ width: `${highPct}%` }} />}
          {medPct > 0 && <div className="bg-ochre transition-all" style={{ width: `${medPct}%` }} />}
          {lowPct > 0 && <div className="bg-forest transition-all" style={{ width: `${lowPct}%` }} />}
        </div>
        <div className="flex justify-between text-[10px] text-slate-400 font-mono">
          <span>고위험 {highPct}%</span>
          <span>중위험 {medPct}%</span>
          <span>저위험 {lowPct}%</span>
        </div>
      </div>
    </div>
  );
}

// ── Analysis progress (6-agent 파이프라인) ─────────────
const ANALYSIS_STEPS = [
  "조항 1차 분석 (Analysis)",
  "관련 법령·판례 검색 (Retrieval Strategy)",
  "근거 재랭킹 (Evidence Selection)",
  "위험도 판단 (Judgment)",
  "유사 사례 교차검증 (Red-team)",
  "근거 충분성 확인 (Evidence Verification)",
];

function AnalysisProgress() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setStep((s) => (s + 1) % ANALYSIS_STEPS.length);
    }, 1400);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="border border-rule bg-white p-4 space-y-2.5">
      <p className="text-[11px] font-mono uppercase tracking-wide text-navy">
        6-Agent 파이프라인 진행 중
      </p>
      <ul className="space-y-1.5">
        {ANALYSIS_STEPS.map((label, i) => (
          <li
            key={label}
            className={`flex items-center gap-2 text-xs transition-colors ${
              i === step ? "text-navy font-semibold" : i < step ? "text-slate-400" : "text-slate-300"
            }`}
          >
            {i < step ? (
              <CheckCircle2 className="h-3 w-3 shrink-0" aria-hidden />
            ) : i === step ? (
              <Loader2 className="h-3 w-3 shrink-0 animate-spin" aria-hidden />
            ) : (
              <span className="h-3 w-3 shrink-0 rounded-full border border-slate-300" aria-hidden />
            )}
            {label}
          </li>
        ))}
      </ul>
      <p className="text-[10px] text-slate-400 pt-1 border-t border-rule">
        조항마다 이 과정을 순차적으로 거칩니다. 조항 수가 많을수록 시간이 더 걸릴 수 있습니다.
      </p>
    </div>
  );
}

// ── Main component ────────────────────────────────────
export function ContractAnalyzer() {
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState<string | null>(null);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [filter, setFilter] = useState<"all" | RiskLevel>("all");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const mutation = useMutation<FullAnalyzeResult, Error, { text: string; pdf?: File }>({
    mutationFn: async ({ text: t, pdf }) => {
      if (pdf) {
        const form = new FormData();
        form.append("file", pdf);
        const res = await fetch("/api/analyze-pdf", {
          method: "POST",
          body: form,
        });
        if (!res.ok) throw new Error("분석에 실패했습니다.");
        return res.json();
      }
      const res = await fetch("/api/analyze-full", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t }),
      });
      if (!res.ok) throw new Error("분석에 실패했습니다.");
      return res.json();
    },
  });

  const handleFile = useCallback((file: File) => {
    if (file.size > MAX_UPLOAD_SIZE_BYTES) {
      alert(`파일 크기는 ${MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)}MB를 초과할 수 없습니다.`);
      return;
    }
    if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
      setPdfFile(file);
      setFileName(file.name);
      setText("");
      mutation.reset();
      return;
    }
    if (!file.type.startsWith("text/") && !file.name.endsWith(".txt")) {
      alert("PDF 또는 .txt 파일을 업로드해 주세요.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setText(content);
      setPdfFile(null);
      setFileName(file.name);
      mutation.reset();
    };
    reader.readAsText(file, "utf-8");
  }, [mutation]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleAnalyze = () => {
    if (pdfFile) {
      mutation.mutate({ text: "", pdf: pdfFile });
      return;
    }
    if (!text.trim()) {
      document.getElementById("contract-input")?.focus();
      return;
    }
    mutation.mutate({ text });
  };

  const filteredClauses = mutation.data?.clauses.filter(
    (c) => filter === "all" || c.risk_level === filter
  ) ?? [];
  const selectedClause =
    filteredClauses.find((c) => c.id === selectedId) ?? filteredClauses[0];

  return (
    <div className="space-y-6">
      {/* ── Input area ── */}
      {!mutation.data && (
        <div className="space-y-4 max-w-2xl mx-auto w-full">
          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed transition-all duration-200 cursor-pointer
              ${dragOver
                ? "border-navy bg-navy-soft"
                : "border-slate-300 hover:border-navy/50 bg-white"
              }`}
            onClick={() => fileRef.current?.click()}
            role="button"
            aria-label="계약서 파일 업로드 영역"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.pdf,text/plain,application/pdf"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
            <div className="flex flex-col items-center gap-3 py-10 px-6 text-center pointer-events-none">
              <div className="p-3 bg-slate-100 rounded-xl">
                <Upload className="h-6 w-6 text-slate-500" aria-hidden />
              </div>
              {fileName ? (
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-navy">
                    <FileText className="h-4 w-4" aria-hidden />
                    <span className="text-sm font-medium">{fileName}</span>
                  </div>
                  <p className="text-xs text-slate-500">{pdfFile ? "PDF 업로드 완료" : `${text.length.toLocaleString()}자 로드됨`}</p>
                </div>
              ) : (
                <div className="space-y-1">
                  <p className="text-sm text-slate-700 font-medium">
                    계약서 파일을 드래그하거나 클릭하여 업로드
                  </p>
                  <p className="text-xs text-slate-400">지원 형식: .pdf · .txt</p>
                </div>
              )}
            </div>
          </div>

          {/* Divider */}
          <div className="flex items-center gap-3">
            <div className="flex-1 h-px bg-slate-200" />
            <span className="text-xs text-slate-400">또는 직접 붙여넣기</span>
            <div className="flex-1 h-px bg-slate-200" />
          </div>

          {/* Textarea */}
          <div className="relative">
            <textarea
              id="contract-input"
              value={text}
              onChange={(e) => { setText(e.target.value); setFileName(null); mutation.reset(); }}
              placeholder="계약서 전문을 붙여넣으세요. AI가 조항별로 자동 분리하여 각각 분석합니다."
              aria-label="분석할 계약서 전문 입력"
              className="w-full min-h-[220px] bg-white border border-slate-300 focus:border-navy text-slate-800 placeholder:text-slate-400 text-sm resize-none rounded-xl px-4 py-3 focus:outline-none transition-colors leading-relaxed"
            />
            {text && (
              <button
                onClick={() => { setText(""); setFileName(null); mutation.reset(); }}
                aria-label="입력 내용 초기화"
                className="absolute top-3 right-3 text-slate-400 hover:text-slate-600 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            )}
            <div className="absolute bottom-3 right-3 text-xs text-slate-400 select-none">
              {text.length.toLocaleString()}자
            </div>
          </div>

          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            disabled={mutation.isPending}
            aria-label="계약서 전체 분석 시작"
            className="w-full h-12 bg-navy hover:opacity-90 disabled:opacity-60 text-white font-semibold rounded-xl transition-opacity flex items-center justify-center gap-2"
          >
            {mutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                조항별 분석 중...
              </>
            ) : (
              <>
                <Search className="h-4 w-4" aria-hidden />
                계약서 전체 분석 시작
              </>
            )}
          </button>

          {mutation.isPending && <AnalysisProgress />}

          {mutation.isError && (
            <p role="alert" className="text-sm text-seal text-center">
              {mutation.error?.message ?? "분석 중 오류가 발생했습니다. 다시 시도해 주세요."}
            </p>
          )}
        </div>
      )}

      {/* ── Results ── */}
      {mutation.data && (
        <div className="space-y-5">
          {/* Reset / export buttons */}
          <div className="flex items-center justify-between print:hidden">
            <h2 className="text-slate-900 font-semibold flex items-center gap-2">
              <Zap className="h-4 w-4 text-navy" aria-hidden />
              분석 결과
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => window.print()}
                className="flex items-center gap-1.5 text-xs text-navy border border-navy/30 hover:bg-navy-soft rounded-lg px-3 py-1.5 transition-colors"
              >
                <Printer className="h-3.5 w-3.5" aria-hidden />
                리포트 저장 (PDF)
              </button>
              <button
                onClick={() => { mutation.reset(); setText(""); setFileName(null); setFilter("all"); setSelectedId(null); }}
                className="text-xs text-slate-500 hover:text-slate-700 border border-slate-300 hover:border-slate-400 rounded-lg px-3 py-1.5 transition-colors"
              >
                새 계약서 분석
              </button>
            </div>
          </div>

          {/* Print-only report header */}
          <div className="hidden print:block">
            <p className="font-mono text-[10px] tracking-wide text-navy uppercase">ContractRisk Guard · AI 리스크 분석 리포트</p>
            <h1 className="text-2xl font-bold text-slate-900 mt-1">계약서 분석 리포트</h1>
            <p className="text-xs text-slate-500 mt-1 font-sans">
              생성일시: {new Date().toLocaleString("ko-KR")} · 본 리포트는 참고용이며 법적 조언을 대체하지 않습니다.
            </p>
          </div>

          {/* Summary */}
          <SummaryBar result={mutation.data} />

          {/* Filter tabs */}
          <div className="flex gap-2 print:hidden" role="tablist" aria-label="위험도 필터">
            {([
              ["all", "전체", mutation.data.total_clauses],
              ["High", "고위험", mutation.data.high_count],
              ["Medium", "중위험", mutation.data.medium_count],
              ["Low", "저위험", mutation.data.low_count],
            ] as [string, string, number][]).map(([val, label, count]) => (
              <button
                key={val}
                role="tab"
                aria-selected={filter === val}
                onClick={() => setFilter(val as typeof filter)}
                className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                  filter === val
                    ? "bg-navy text-white"
                    : "bg-white text-slate-600 border border-rule hover:bg-navy-soft"
                }`}
              >
                {label} <span className="ml-1 opacity-70">{count}</span>
              </button>
            ))}
          </div>

          {/* Risk minimap */}
          <RiskMinimap
            clauses={filteredClauses}
            selectedId={selectedClause?.id}
            onSelect={setSelectedId}
          />

          {/* Clause list + detail (screen only) */}
          {filteredClauses.length === 0 ? (
            <p className="text-center text-slate-400 py-8 text-sm print:hidden">해당 위험도의 조항이 없습니다.</p>
          ) : (
            <div
              data-testid="clause-workspace"
              className="md:grid md:grid-cols-[220px_1fr] md:gap-5 md:items-start print:hidden"
            >
              {/* Sidebar: clause list */}
              <nav
                aria-label="분석된 조항 목록"
                className="flex md:flex-col gap-1.5 overflow-x-auto md:overflow-x-visible mb-4 md:mb-0 pb-2 md:pb-0 md:max-h-[600px] md:overflow-y-auto md:border md:border-rule md:p-2 md:bg-white"
              >
                {filteredClauses.map((clause) => (
                  <ClauseListItem
                    key={clause.id}
                    clause={clause}
                    active={selectedClause?.id === clause.id}
                    onClick={() => setSelectedId(clause.id)}
                  />
                ))}
              </nav>

              {/* Detail panel */}
              {selectedClause && <ClauseDetail clause={selectedClause} />}
            </div>
          )}

          {/* Full clause list (print only) */}
          <div className="hidden print:block space-y-6">
            {filteredClauses.map((clause) => (
              <ClauseDetail key={clause.id} clause={clause} />
            ))}
          </div>

          <p className="text-xs text-slate-400 text-center pb-2 print:hidden">
            최대 20개 조항까지 분석됩니다. 전체 조항 분석은 순차 확장 예정입니다.
          </p>
        </div>
      )}
    </div>
  );
}
