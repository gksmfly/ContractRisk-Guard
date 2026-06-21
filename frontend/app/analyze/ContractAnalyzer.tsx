// frontend/app/analyze/ContractAnalyzer.tsx
"use client";

import { useState, useRef, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import type { FullAnalyzeResult } from "@/app/api/analyze-full/route";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileText,
  Loader2,
  Search,
  Shield,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";

// ── Types ──────────────────────────────────────────────
type RiskLevel = "High" | "Medium" | "Low";

const RISK_CFG: Record<
  RiskLevel,
  { label: string; color: string; bg: string; border: string; badgeCls: string; icon: typeof AlertTriangle }
> = {
  High: {
    label: "고위험",
    color: "text-red-600",
    bg: "bg-red-50",
    border: "border-red-200",
    badgeCls: "bg-red-100 text-red-700 border-red-200",
    icon: AlertTriangle,
  },
  Medium: {
    label: "중위험",
    color: "text-amber-600",
    bg: "bg-amber-50",
    border: "border-amber-200",
    badgeCls: "bg-amber-100 text-amber-700 border-amber-200",
    icon: AlertTriangle,
  },
  Low: {
    label: "저위험",
    color: "text-emerald-600",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    badgeCls: "bg-emerald-100 text-emerald-700 border-emerald-200",
    icon: CheckCircle2,
  },
};

// ── Highlight helper ───────────────────────────────────
function HighlightText({
  text,
  spans,
}: {
  text: string;
  spans: { text: string; start: number; end: number }[];
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
          <mark key={i} className="bg-amber-100 text-amber-800 rounded px-0.5 not-italic font-medium">
            {p.t}
          </mark>
        ) : (
          <span key={i} className="text-slate-700">{p.t}</span>
        )
      )}
    </span>
  );
}

// ── Clause card ───────────────────────────────────────
function ClauseCard({ clause }: { clause: FullAnalyzeResult["clauses"][0] }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = RISK_CFG[clause.risk_level];
  const Icon = cfg.icon;

  return (
    <article className={`border ${cfg.border} rounded-xl overflow-hidden`}>
      {/* Header row */}
      <div className={`${cfg.bg} px-4 py-3 flex items-center justify-between gap-3`}>
        <div className="flex items-center gap-3 min-w-0">
          <div className={`p-1.5 rounded-lg bg-white/70 shrink-0`}>
            <Icon className={`h-4 w-4 ${cfg.color}`} aria-hidden />
          </div>
          <span className="text-xs text-slate-500 shrink-0">제{clause.id}항</span>
          <span className={`text-xs font-mono ${cfg.color} shrink-0`}>
            {clause.domain === "해당없음" ? "분류 불가" : clause.domain}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Badge className={`${cfg.badgeCls} border text-xs`}>{cfg.label}</Badge>
          <button
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
            aria-label={`제${clause.id}항 상세 ${expanded ? "접기" : "펼치기"}`}
            className="text-slate-400 hover:text-slate-700 transition-colors p-1"
          >
            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Original text */}
      <div className="px-4 py-3 bg-white">
        {clause.evidence_spans.length > 0 ? (
          <HighlightText text={clause.original} spans={clause.evidence_spans} />
        ) : (
          <p className="text-slate-600 text-sm leading-relaxed">{clause.original}</p>
        )}
      </div>

      {/* Detail */}
      {expanded && (
        <div className="px-4 py-4 space-y-3 border-t border-slate-200 bg-slate-50/50">
          {/* Legal basis */}
          {clause.legal_basis.length > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs text-slate-500">
                <Shield className="h-3 w-3" aria-hidden />
                <span>적용 법령</span>
              </div>
              {clause.legal_basis.map((lb, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 bg-white border border-slate-200 rounded-lg p-2.5 text-xs"
                >
                  <span className="text-blue-600 font-mono font-semibold shrink-0">
                    {lb.law} {lb.article}
                  </span>
                  <span className="text-slate-600">{lb.description}</span>
                </div>
              ))}
            </div>
          )}

          {/* Reasoning */}
          {clause.reasoning && (
            <p className="text-xs text-slate-600 bg-blue-50 border border-blue-100 rounded-lg p-3 leading-relaxed">
              {clause.reasoning}
            </p>
          )}
        </div>
      )}
    </article>
  );
}

// ── Summary bar ───────────────────────────────────────
function SummaryBar({ result }: { result: FullAnalyzeResult }) {
  const highPct = Math.round((result.high_count / result.total_clauses) * 100);
  const medPct = Math.round((result.medium_count / result.total_clauses) * 100);
  const lowPct = 100 - highPct - medPct;

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 shadow-sm">
      <div className="flex flex-wrap gap-6 justify-between">
        <div>
          <p className="text-slate-500 text-xs mb-1">총 분석 조항</p>
          <p className="text-3xl font-extrabold text-slate-900">{result.total_clauses}<span className="text-lg text-slate-400 ml-1">건</span></p>
        </div>
        <div className="flex gap-5">
          {[
            { label: "고위험", count: result.high_count, color: "text-red-600" },
            { label: "중위험", count: result.medium_count, color: "text-amber-600" },
            { label: "저위험", count: result.low_count, color: "text-emerald-600" },
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
        <p className="text-xs text-slate-500">위험도 분포</p>
        <div className="flex h-2 rounded-full overflow-hidden gap-px bg-slate-100">
          {highPct > 0 && <div className="bg-red-500 transition-all" style={{ width: `${highPct}%` }} />}
          {medPct > 0 && <div className="bg-amber-500 transition-all" style={{ width: `${medPct}%` }} />}
          {lowPct > 0 && <div className="bg-emerald-500 transition-all" style={{ width: `${lowPct}%` }} />}
        </div>
        <div className="flex justify-between text-[10px] text-slate-400">
          <span>고위험 {highPct}%</span>
          <span>중위험 {medPct}%</span>
          <span>저위험 {lowPct}%</span>
        </div>
      </div>
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
  );

  return (
    <div className="space-y-6">
      {/* ── Input area ── */}
      {!mutation.data && (
        <div className="space-y-4">
          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-2xl transition-all duration-200 cursor-pointer
              ${dragOver
                ? "border-blue-500 bg-blue-50"
                : "border-slate-300 hover:border-blue-400 bg-white"
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
                  <div className="flex items-center gap-2 text-blue-600">
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
              className="w-full min-h-[220px] bg-white border border-slate-300 focus:border-blue-500 text-slate-800 placeholder:text-slate-400 text-sm resize-none rounded-xl px-4 py-3 focus:outline-none transition-colors leading-relaxed"
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
            className="w-full h-12 bg-blue-600 hover:bg-blue-500 disabled:opacity-60 text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2"
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

          {mutation.isError && (
            <p role="alert" className="text-sm text-red-600 text-center">
              {mutation.error?.message ?? "분석 중 오류가 발생했습니다. 다시 시도해 주세요."}
            </p>
          )}
        </div>
      )}

      {/* ── Results ── */}
      {mutation.data && (
        <div className="space-y-5">
          {/* Reset button */}
          <div className="flex items-center justify-between">
            <h2 className="text-slate-900 font-semibold flex items-center gap-2">
              <Zap className="h-4 w-4 text-blue-600" aria-hidden />
              분석 결과
            </h2>
            <button
              onClick={() => { mutation.reset(); setText(""); setFileName(null); setFilter("all"); }}
              className="text-xs text-slate-500 hover:text-slate-700 border border-slate-300 hover:border-slate-400 rounded-lg px-3 py-1.5 transition-colors"
            >
              새 계약서 분석
            </button>
          </div>

          {/* Summary */}
          <SummaryBar result={mutation.data} />

          {/* Filter tabs */}
          <div className="flex gap-2" role="tablist" aria-label="위험도 필터">
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
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  filter === val
                    ? "bg-blue-600 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {label} <span className="ml-1 opacity-70">{count}</span>
              </button>
            ))}
          </div>

          {/* Clause list */}
          <div className="space-y-3">
            {filteredClauses?.length === 0 ? (
              <p className="text-center text-slate-400 py-8 text-sm">해당 위험도의 조항이 없습니다.</p>
            ) : (
              filteredClauses?.map((clause) => (
                <ClauseCard key={clause.id} clause={clause} />
              ))
            )}
          </div>

          <p className="text-xs text-slate-400 text-center pb-2">
            최대 20개 조항까지 분석됩니다. 전체 조항 분석은 순차 확장 예정입니다.
          </p>
        </div>
      )}
    </div>
  );
}
