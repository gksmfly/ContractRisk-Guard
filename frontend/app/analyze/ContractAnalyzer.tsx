// frontend/app/analyze/ContractAnalyzer.tsx
"use client";

import { useState, useRef, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import type { FullAnalyzeResult } from "@/app/api/analyze-full/route";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileText,
  HighlighterIcon,
  Loader2,
  Printer,
  Scale,
  Search,
  Share2,
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
      <span className="text-xs text-slate-400 font-mono shrink-0">§{clause.id}</span>
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
  boxed,
}: {
  clauses: Clause[];
  selectedId: number | undefined;
  onSelect: (id: number) => void;
  boxed?: boolean;
}) {
  if (clauses.length <= 1) return null;

  const strip = (
    <div role="group" aria-label="조항별 위험도 미니맵" className="flex items-center gap-1.5 flex-wrap">
      {clauses.map((c) => {
        const cfg = RISK_CFG[c.risk_level];
        const active = c.id === selectedId;
        return (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            aria-label={`§${c.id}(${cfg.label})으로 이동`}
            aria-current={active}
            title={`§${c.id} · ${cfg.label}`}
            className={`h-5 w-5 shrink-0 transition-all ${cfg.dotCls} ${
              active ? "ring-2 ring-offset-1 ring-navy" : "opacity-60 hover:opacity-100"
            }`}
          />
        );
      })}
    </div>
  );

  if (!boxed) return <div className="print:hidden">{strip}</div>;

  return (
    <div className="space-y-2 print:hidden">
      <p className="text-[11px] font-mono uppercase tracking-wide text-slate-500">
        전체 조항 위험도 지도
      </p>
      <div className="border border-rule bg-white p-3">{strip}</div>
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
    <div className="border border-rule bg-white p-3.5 print:break-inside-avoid">
      <div className="flex items-center gap-1.5 mb-1.5">
        <Scale className="h-3 w-3 text-navy shrink-0" aria-hidden />
        <span className="text-xs font-mono font-semibold text-navy truncate">
          {basis.law} {basis.article}
        </span>
      </div>
      <blockquote className="text-[13px] text-slate-700 leading-relaxed">
        “{basis.description}”
      </blockquote>
      {onJumpToEvidence && (
        <button
          onClick={onJumpToEvidence}
          className="mt-2.5 flex items-center gap-1 text-[11px] font-medium text-navy bg-navy-soft hover:bg-navy hover:text-white rounded-full px-2.5 py-1 transition-colors print:hidden"
        >
          <HighlighterIcon className="h-3 w-3" aria-hidden />
          원문에서 보기
        </button>
      )}
    </div>
  );
}

// ── Per-clause text export (정적/데모 범위 — 실제 다운로드는 됨) ──
function exportClauseAsText(clause: Clause) {
  const cfg = RISK_CFG[clause.risk_level];
  const lines = [
    `Verilex — 조항 §${clause.id} 분석 결과`,
    `도메인: ${clause.domain} · 위험도: ${cfg.label}`,
    "",
    "[원문]",
    clause.original,
    "",
    "[판단 근거]",
    clause.reasoning || "-",
    "",
    "[적용 법령 원문]",
    ...clause.legal_basis.map((lb) => `${lb.law} ${lb.article}\n"${lb.description}"`),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `조항-${clause.id}-분석결과.txt`;
  a.click();
  URL.revokeObjectURL(url);
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
function ClauseDetail({
  clause,
  allClauses,
  onSelectClause,
}: {
  clause: Clause;
  allClauses?: Clause[];
  onSelectClause?: (id: number) => void;
}) {
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
        <span>§{clause.id} · AI 리스크 판단 결과</span>
        <span className="flex items-center gap-2">
          {clause.verified && (
            <span
              className="inline-flex items-center gap-1 bg-forest/20 text-forest-soft rounded-full px-2 py-0.5"
              aria-label="Red-team 검증 통과"
            >
              <CheckCircle2 className="h-3 w-3" aria-hidden /> 검증됨
            </span>
          )}
          {!clause.evidence_verified && (
            <span
              className="bg-white/10 rounded-full px-2 py-0.5"
              aria-label="근거 재검색이 필요했던 조항"
            >
              근거 미확정
            </span>
          )}
        </span>
      </div>

      {/* Header row */}
      <div className="px-5 py-4 flex items-start justify-between gap-4 border-b border-rule">
        <div className="min-w-0">
          <p className="text-[11px] font-mono text-slate-500">
            {clause.domain === "해당없음" ? "분류 불가" : clause.domain}
          </p>
          <p className="text-lg font-semibold text-slate-900 mt-0.5">
            위험도 판단: {cfg.label}
          </p>
        </div>
        <div
          className={`shrink-0 w-16 h-16 rounded-full ${cfg.dotCls} text-white flex items-center justify-center font-mono shadow-sm`}
          aria-hidden
        >
          <span className="text-xs font-bold leading-none text-center">{cfg.label}</span>
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
        {/* Judgment summary */}
        {clause.reasoning && (
          <div className={`border ${cfg.ring} ${cfg.bg} p-4`}>
            <p className={`text-xs font-bold ${cfg.color} mb-1.5`}>판단 요약</p>
            <p className="text-sm text-slate-700 leading-relaxed">{clause.reasoning}</p>
          </div>
        )}

        {/* Legal basis */}
        {clause.legal_basis.length > 0 && (
          <div className="space-y-2">
            <SectionLabel icon={Shield}>적용 법령 원문</SectionLabel>
            <div className="space-y-2">
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

        {/* Red-team note */}
        {clause.redteam_note && (
          <div className="space-y-2">
            <SectionLabel icon={AlertTriangle}>Red-team 검토 의견</SectionLabel>
            <p className={`text-xs text-slate-600 ${cfg.bg} border-l-2 ${cfg.ring} px-3.5 py-2.5 leading-relaxed`}>
              {clause.redteam_note}
            </p>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 print:hidden">
          <button
            onClick={() => exportClauseAsText(clause)}
            className="flex items-center gap-1.5 text-xs font-medium text-slate-600 border border-rule hover:bg-slate-50 rounded-lg px-3 py-1.5 transition-colors"
          >
            <Download className="h-3.5 w-3.5" aria-hidden />
            이 조항만 내보내기
          </button>
        </div>

        {/* Risk map (전체 조항 위험도 지도) */}
        {allClauses && onSelectClause && (
          <RiskMinimap clauses={allClauses} selectedId={clause.id} onSelect={onSelectClause} boxed />
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

// ── Analysis progress (조항 단위 실시간 진행률) ─────────
// 백엔드가 조항을 순차 처리하는 동안 SSE로 "조항 N/M 완료" 이벤트를 실제로
// 흘려보낸다 — 그래프 내부(Analysis→Judgment/Retrieval fan-out, 재검색 루프)는
// 조항 하나 안에서 병렬·가변 경로라 에이전트 단계별로는 실시간 신호를 얻을 수
// 없어서, "조항이 몇 개 중 몇 개 끝났는지"를 진행률의 최소 단위로 쓴다.
export interface StreamProgress {
  index: number; // 지금까지 완료된 조항 수 (1-based)
  total: number;
}

// B2B: 조항별 완료 리스트
function AgentListProgress({ progress }: { progress: StreamProgress | null }) {
  if (!progress) {
    return (
      <div className="border border-rule bg-white px-4 py-6 text-center text-xs text-slate-400">
        조항을 분리하고 있습니다...
      </div>
    );
  }

  const { index: completed, total } = progress;
  const rows = Array.from({ length: total }, (_, i) => i + 1);

  return (
    <div className="border border-rule bg-white divide-y divide-rule text-left">
      {rows.map((n) => {
        const status = n <= completed ? "done" : n === completed + 1 ? "active" : "waiting";
        return (
          <div
            key={n}
            className={`flex items-center gap-3 px-4 py-2.5 transition-colors ${
              status === "active" ? "bg-navy-soft" : ""
            }`}
          >
            {status === "done" ? (
              <span className="h-5 w-5 shrink-0 rounded-full bg-forest text-white flex items-center justify-center">
                <CheckCircle2 className="h-3 w-3" aria-hidden />
              </span>
            ) : status === "active" ? (
              <span className="h-5 w-5 shrink-0 rounded-full bg-navy text-white flex items-center justify-center">
                <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
              </span>
            ) : (
              <span className="h-5 w-5 shrink-0 rounded-full border border-slate-300 text-slate-400 text-[10px] flex items-center justify-center">
                {n}
              </span>
            )}
            <div className="min-w-0 flex-1">
              <p className={`text-xs font-semibold ${status !== "waiting" ? "text-slate-900" : "text-slate-400"}`}>
                조항 {n}
              </p>
              <p className="text-[11px] text-slate-500 truncate">
                {status === "done" ? "6-agent 파이프라인 완료" : status === "active" ? "분석·검색·판정·검증 진행 중" : "대기 중"}
              </p>
            </div>
            {status === "done" && (
              <span className="shrink-0 text-[10px] font-medium bg-forest-soft text-forest rounded-full px-2 py-0.5">완료</span>
            )}
            {status === "active" && (
              <span className="shrink-0 text-[10px] font-medium bg-navy text-white rounded-full px-2 py-0.5">진행 중</span>
            )}
          </div>
        );
      })}
      <p className="text-[10px] text-slate-400 px-4 py-2">
        조항마다 Analysis→Retrieval→Judgment→Red-team→Evidence Verification 파이프라인을 거칩니다.
      </p>
    </div>
  );
}

// B2C: 축약형 진행바
function CompactProgress({ progress }: { progress: StreamProgress | null }) {
  const pct = progress ? Math.round((progress.index / progress.total) * 100) : 0;

  return (
    <div className="space-y-3">
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full bg-navy transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      <p className="text-xs text-slate-400">
        {progress ? `조항 ${progress.index}/${progress.total}개 분석 완료` : "조항을 분리하고 있습니다..."}
      </p>
    </div>
  );
}

function AnalysisProgress({ mode, progress }: { mode: "b2c" | "b2b"; progress: StreamProgress | null }) {
  return mode === "b2b" ? <AgentListProgress progress={progress} /> : <CompactProgress progress={progress} />;
}

// ── Step breadcrumb (약관 업로드 → 분석 중 → 결과 확인) ──
function StepBreadcrumb({ active }: { active: "upload" | "analyzing" }) {
  const steps: { key: "upload" | "analyzing" | "result"; label: string }[] = [
    { key: "upload", label: "약관 업로드" },
    { key: "analyzing", label: "분석 중" },
    { key: "result", label: "결과 확인" },
  ];
  const activeIdx = steps.findIndex((s) => s.key === active);

  return (
    <nav aria-label="분석 진행 단계" className="flex items-center justify-center gap-2 text-xs">
      {steps.map((s, i) => (
        <span key={s.key} className="flex items-center gap-2">
          <span
            aria-current={i === activeIdx ? "step" : undefined}
            className={
              i === activeIdx
                ? "font-semibold text-navy"
                : i < activeIdx
                ? "text-slate-500"
                : "text-slate-300"
            }
          >
            {s.label}
          </span>
          {i < steps.length - 1 && <span className="text-slate-300" aria-hidden>→</span>}
        </span>
      ))}
    </nav>
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
  const [mode, setMode] = useState<"b2c" | "b2b">("b2c");
  const [inputTab, setInputTab] = useState<"text" | "pdf">("text");
  const [streamProgress, setStreamProgress] = useState<StreamProgress | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // /api/analyze-stream, /api/analyze-pdf-stream이 흘려보내는 SSE("data: {...}\n\n")를
  // 읽어 조항이 끝날 때마다 onProgress를 부르고, 마지막 done 이벤트의 결과를 반환한다.
  const consumeAnalyzeStream = useCallback(
    async (res: Response, onProgress: (p: StreamProgress) => void): Promise<FullAnalyzeResult> => {
      if (!res.ok || !res.body) throw new Error("분석에 실패했습니다.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let result: FullAnalyzeResult | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";
        for (const chunk of chunks) {
          const line = chunk.trim();
          if (!line.startsWith("data:")) continue;
          const event = JSON.parse(line.slice(5).trim());
          if (event.type === "progress") {
            onProgress({ index: event.index, total: event.total });
          } else if (event.type === "done") {
            result = event.result;
          }
        }
      }

      if (!result) throw new Error("분석 서버 응답이 올바르지 않습니다.");
      return result;
    },
    []
  );

  const mutation = useMutation<FullAnalyzeResult, Error, { text: string; pdf?: File }>({
    mutationFn: async ({ text: t, pdf }) => {
      setStreamProgress(null);

      if (pdf) {
        const form = new FormData();
        form.append("file", pdf);
        const res = await fetch("/api/analyze-pdf-stream", { method: "POST", body: form });
        return consumeAnalyzeStream(res, setStreamProgress);
      }

      const res = await fetch("/api/analyze-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t }),
      });
      return consumeAnalyzeStream(res, setStreamProgress);
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
        <div className="space-y-5 max-w-2xl mx-auto w-full">
          <StepBreadcrumb active={mutation.isPending ? "analyzing" : "upload"} />

          {mutation.isPending ? (
            <div className="py-6 space-y-5 text-center">
              <div className="w-16 h-16 mx-auto rounded-full bg-navy-soft flex items-center justify-center">
                <Loader2 className="h-6 w-6 text-navy animate-spin" aria-hidden />
              </div>
              <div className="space-y-1.5">
                <h1 className="text-xl font-bold text-slate-900">
                  {mode === "b2b" ? "법령 근거를 검토하고 있습니다" : "약관을 분석하고 있습니다"}
                </h1>
                <p className="text-sm text-slate-500">
                  {mode === "b2b"
                    ? "6개 AI 에이전트가 법령 근거를 검토하고 판단합니다. 잠시만 기다려주세요."
                    : "약관규제법 조문 검색 중 · 판례 대조 중 · 위험도 판정 중"}
                </p>
              </div>
              <AnalysisProgress mode={mode} progress={streamProgress} />
            </div>
          ) : (
            <>
          <h1 className="text-2xl font-bold text-slate-900 text-center">
            약관을 붙여넣거나 파일로 업로드하세요
          </h1>

          {/* Input tabs */}
          <div className="flex justify-center gap-1 border-b border-rule" role="tablist" aria-label="입력 방식">
            {([
              ["text", "텍스트 붙여넣기"],
              ["pdf", "PDF 업로드"],
            ] as const).map(([val, label]) => (
              <button
                key={val}
                role="tab"
                aria-selected={inputTab === val}
                onClick={() => setInputTab(val)}
                className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                  inputTab === val
                    ? "border-navy text-navy"
                    : "border-transparent text-slate-400 hover:text-slate-600"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {inputTab === "text" ? (
            /* Textarea */
            <div className="relative">
              <textarea
                id="contract-input"
                value={text}
                onChange={(e) => { setText(e.target.value); setFileName(null); mutation.reset(); }}
                placeholder={'여기에 약관 전문을 붙여넣으세요.\n\n예) 제1조 (목적) 이 약관은 ○○주식회사(이하 "회사")가 운영하는 서비스의 이용과 관련하여...'}
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
          ) : (
            /* PDF drop zone */
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-xl transition-all duration-200 cursor-pointer
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
                accept=".pdf,application/pdf"
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
                    <p className="text-xs text-slate-500">PDF 업로드 완료</p>
                  </div>
                ) : (
                  <div className="space-y-1">
                    <p className="text-sm text-slate-700 font-medium">
                      파일을 여기에 끌어다 놓거나 클릭하여 선택
                    </p>
                    <p className="text-xs text-slate-400">PDF · 최대 {MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)}MB</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Mode toggle */}
          <div className="flex gap-2 justify-center" role="radiogroup" aria-label="분석 모드">
            {([
              ["b2c", "B2C 소비자 모드 (요약 판정)"],
              ["b2b", "B2B 전문 모드 (전체 분석 + PDF)"],
            ] as const).map(([val, label]) => (
              <button
                key={val}
                role="radio"
                aria-checked={mode === val}
                onClick={() => setMode(val)}
                className={`flex items-center gap-2 text-xs font-medium px-3.5 py-2 rounded-lg border transition-colors ${
                  mode === val
                    ? "border-navy bg-navy-soft text-navy"
                    : "border-slate-200 text-slate-500 hover:border-slate-300"
                }`}
              >
                <span
                  className={`h-2.5 w-2.5 rounded-full border-2 shrink-0 ${
                    mode === val ? "border-navy bg-navy" : "border-slate-300"
                  }`}
                  aria-hidden
                />
                {label}
              </button>
            ))}
          </div>

          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            aria-label="계약서 전체 분석 시작"
            className="w-full h-12 bg-navy hover:opacity-90 text-white font-semibold rounded-xl transition-opacity flex items-center justify-center gap-2"
          >
            <Search className="h-4 w-4" aria-hidden />
            AI 분석 시작하기
          </button>

          {mutation.isError && (
            <p role="alert" className="text-sm text-seal text-center">
              {mutation.error?.message ?? "분석 중 오류가 발생했습니다. 다시 시도해 주세요."}
            </p>
          )}
            </>
          )}
        </div>
      )}

      {/* ── Results ── */}
      {mutation.data && (
        <div className="space-y-5">
          {/* Reset / export buttons */}
          <div className="flex items-center justify-between gap-3 flex-wrap print:hidden">
            <div className="flex items-center gap-2 min-w-0">
              <Zap className="h-4 w-4 text-navy shrink-0" aria-hidden />
              <h2 className="text-slate-900 font-semibold truncate">
                {fileName ?? "분석 결과"}
              </h2>
              <span className="shrink-0 bg-forest-soft text-forest text-[11px] font-medium rounded-full px-2 py-0.5">
                분석 완료
              </span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {mode === "b2b" && (
                <button
                  onClick={() => window.print()}
                  title="인쇄 대화상자가 열리면 대상(프린터)을 'PDF로 저장'으로 선택하세요"
                  aria-label="PDF 리포트 — 인쇄 대화상자에서 'PDF로 저장'을 선택하세요"
                  className="flex items-center gap-1.5 text-xs text-navy border border-navy/30 hover:bg-navy-soft rounded-lg px-3 py-1.5 transition-colors"
                >
                  <Printer className="h-3.5 w-3.5" aria-hidden />
                  PDF 리포트
                </button>
              )}
              <button
                onClick={async () => {
                  const shareData = { title: "Verilex 분석 결과", url: window.location.href };
                  if (navigator.share) {
                    await navigator.share(shareData).catch(() => {});
                  } else {
                    await navigator.clipboard.writeText(shareData.url);
                    alert("링크를 클립보드에 복사했습니다.");
                  }
                }}
                className="flex items-center gap-1.5 text-xs bg-navy hover:opacity-90 text-white rounded-lg px-3 py-1.5 transition-opacity"
              >
                <Share2 className="h-3.5 w-3.5" aria-hidden />
                공유
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
            <p className="font-mono text-[10px] tracking-wide text-navy uppercase">Verilex · AI 리스크 분석 리포트</p>
            <h1 className="text-2xl font-bold text-slate-900 mt-1">계약서 분석 리포트</h1>
            <p className="text-xs text-slate-500 mt-1 font-sans">
              생성일시: {new Date().toLocaleString("ko-KR")} · 본 리포트는 참고용이며 법적 조언을 대체하지 않습니다.
            </p>
          </div>

          {/* Summary */}
          <SummaryBar result={mutation.data} />

          {/* Filter tabs (B2B 전용 — B2C는 요약 판정만 보여주므로 생략) */}
          {mode === "b2b" && (
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
          )}

          {/* 결과 본문: screen 전용 */}
          {filteredClauses.length === 0 ? (
            <p className="text-center text-slate-400 py-8 text-sm print:hidden">해당 위험도의 조항이 없습니다.</p>
          ) : mode === "b2c" ? (
            /* B2C 소비자 모드: 사이드바 없이 요약 판정 카드만 순서대로 */
            <div data-testid="clause-workspace" className="space-y-4 print:hidden">
              {filteredClauses.map((clause) => (
                <ClauseDetail key={clause.id} clause={clause} />
              ))}
            </div>
          ) : (
            /* B2B 전문 모드: 사이드바 + 상세 패널 */
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
              {selectedClause && (
                <ClauseDetail
                  clause={selectedClause}
                  allClauses={filteredClauses}
                  onSelectClause={setSelectedId}
                />
              )}
            </div>
          )}

          {/* Full clause list (print only, B2B 리포트 전용) */}
          {mode === "b2b" && (
            <div className="hidden print:block space-y-6">
              {filteredClauses.map((clause) => (
                <ClauseDetail key={clause.id} clause={clause} />
              ))}
            </div>
          )}

          <p className="text-xs text-slate-400 text-center pb-2 print:hidden">
            최대 20개 조항까지 분석됩니다. 전체 조항 분석은 순차 확장 예정입니다.
          </p>
        </div>
      )}
    </div>
  );
}
