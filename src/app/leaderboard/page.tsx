"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Upload, FileText, X, Loader2, ArrowRight, FileSearch,
  Trophy, ChevronDown, ChevronUp, Info,
} from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

interface JobCard {
  id: number;
  job_title: string | null;
  employer_name: string | null;
  city: string | null;
  province: string | null;
  salary: string | null;
  broad_category: string | null;
  date_posted: string | null;
  jd_preview: string | null;
}

interface ResumeEntry {
  name: string;
  resume_text: string;
}

interface Cell {
  score: number;
  verdict: string;
}

interface BulkRow {
  resume_name: string;
  best_score: number;
  best_job_title: string | null;
  best_job_id: number | null;
  cells: { [jobId: string]: Cell };
}

interface BulkMatchResult {
  resumes: string[];
  jobs: JobCard[];
  rows: BulkRow[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function scoreColor(score: number): { bg: string; text: string } {
  if (score >= 70) return { bg: "var(--success-bg)", text: "var(--success-text)" };
  if (score >= 40) return { bg: "var(--warning-bg)", text: "var(--warning-text)" };
  return { bg: "var(--danger-bg)", text: "var(--danger-text)" };
}

function ScoreBadge({ score, verdict }: { score: number; verdict: string }) {
  const [tip, setTip] = useState(false);
  const { bg, text } = scoreColor(score);
  return (
    <td
      className="px-2 py-2 text-center relative cursor-pointer select-none"
      onMouseEnter={() => setTip(true)}
      onMouseLeave={() => setTip(false)}
    >
      <span
        className="inline-block min-w-[2.5rem] rounded-md px-1.5 py-0.5 text-xs font-bold"
        style={{ background: bg, color: text }}
      >
        {score}
      </span>
      {tip && (
        <div
          className="absolute z-20 bottom-full left-1/2 -translate-x-1/2 mb-1 w-28 rounded-lg p-2 text-xs shadow-lg"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
        >
          {verdict}
        </div>
      )}
    </td>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function LeaderboardPage() {
  const router = useRouter();

  // Phase: "setup" | "loading" | "results"
  const [phase, setPhase] = useState<"setup" | "loading" | "results">("setup");

  // Setup state
  const [selectedJobIds, setSelectedJobIds] = useState<number[]>([]);
  const [selectedJobCards, setSelectedJobCards] = useState<JobCard[]>([]);
  const [resumes, setResumes] = useState<ResumeEntry[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Results state
  const [result, setResult] = useState<BulkMatchResult | null>(null);

  // Load sessionStorage on mount
  useEffect(() => {
    const stored = sessionStorage.getItem("bulkMatchResult");
    if (stored) {
      try {
        setResult(JSON.parse(stored));
        setPhase("results");
        return;
      } catch {
        // ignore parse error, fall through to setup
      }
    }

    const ids = sessionStorage.getItem("selectedJobIds");
    const cards = sessionStorage.getItem("selectedJobCards");
    if (ids) {
      try {
        setSelectedJobIds(JSON.parse(ids));
        if (cards) setSelectedJobCards(JSON.parse(cards));
      } catch {
        // ignore
      }
    }
  }, []);

  // Upload a PDF
  const handleFileUpload = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported");
      return;
    }
    if (resumes.length >= 20) {
      setError("Maximum 20 resumes");
      return;
    }
    if (resumes.some((r) => r.name === file.name)) {
      setError(`"${file.name}" already added`);
      return;
    }

    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE_URL}/api/upload-resume`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      setResumes((prev) => [...prev, { name: data.filename, resume_text: data.text }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [resumes]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    Array.from(e.dataTransfer.files).forEach(handleFileUpload);
  }, [handleFileUpload]);

  const removeResume = (name: string) =>
    setResumes((prev) => prev.filter((r) => r.name !== name));

  // Run bulk match
  const runMatch = async () => {
    if (resumes.length === 0) {
      setError("Upload at least one resume");
      return;
    }
    if (selectedJobIds.length === 0) {
      setError("No jobs selected. Go to Job Search to pick jobs first.");
      return;
    }

    setPhase("loading");
    setError(null);

    try {
      const payload = {
        resumes: resumes.map((r) => ({ name: r.name, resume_text: r.resume_text })),
        job_ids: selectedJobIds,
        max_jobs: Math.min(selectedJobIds.length, 200),
      };

      const res = await fetch(`${API_BASE_URL}/api/bulk-match`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Bulk match failed");
      }

      const data: BulkMatchResult = await res.json();
      sessionStorage.setItem("bulkMatchResult", JSON.stringify(data));
      setResult(data);
      setPhase("results");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Bulk match failed");
      setPhase("setup");
    }
  };

  const resetAndSearch = () => {
    sessionStorage.removeItem("bulkMatchResult");
    sessionStorage.removeItem("selectedJobIds");
    sessionStorage.removeItem("selectedJobCards");
    router.push("/jobs");
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header
        className="flex items-center justify-between px-8 py-4 border-b bg-[var(--bg-card)] flex-shrink-0"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "var(--text-primary)" }}
          >
            <Trophy size={16} style={{ color: "var(--text-on-primary)" }} />
          </div>
          <button
            onClick={() => router.push("/")}
            className="text-lg font-bold tracking-tight hover:opacity-80 transition-opacity"
            style={{ color: "var(--text-primary)" }}
          >
            ResumeMatch
          </button>
          <span className="text-xs ml-1" style={{ color: "var(--text-muted)" }}>/ Leaderboard</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/jobs")}
            className="text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
            style={{ color: "var(--text-secondary)", background: "var(--bg-hover)" }}
          >
            ← Job Search
          </button>
          <ThemeToggle />
        </div>
      </header>

      {/* ── SETUP PHASE ── */}
      {phase === "setup" && (
        <main className="flex-1 overflow-y-auto p-8">
          <div className="max-w-3xl mx-auto space-y-6">
            <div>
              <h1 className="text-2xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
                Bulk Resume Match
              </h1>
              <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
                Upload resumes to match against the {selectedJobIds.length} selected job{selectedJobIds.length !== 1 ? "s" : ""}
              </p>
            </div>

            {/* Selected jobs summary */}
            {selectedJobIds.length > 0 ? (
              <section
                className="rounded-[var(--radius)] border p-4"
                style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
              >
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                    Selected Jobs ({selectedJobIds.length})
                  </h2>
                  <button
                    onClick={resetAndSearch}
                    className="text-xs px-2 py-1 rounded hover:opacity-70"
                    style={{ color: "var(--accent)" }}
                  >
                    Change selection →
                  </button>
                </div>
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {selectedJobCards.length > 0
                    ? selectedJobCards.map((j) => (
                        <div key={j.id} className="flex items-center gap-2 text-xs py-0.5">
                          <span className="font-medium truncate flex-1" style={{ color: "var(--text-primary)" }}>
                            {j.job_title ?? `Job #${j.id}`}
                          </span>
                          {j.city && (
                            <span style={{ color: "var(--text-muted)" }}>{j.city}{j.province ? `, ${j.province}` : ""}</span>
                          )}
                        </div>
                      ))
                    : selectedJobIds.map((id) => (
                        <div key={id} className="text-xs py-0.5" style={{ color: "var(--text-muted)" }}>
                          Job #{id}
                        </div>
                      ))}
                </div>
              </section>
            ) : (
              <div
                className="rounded-[var(--radius)] border p-4 text-center"
                style={{ borderColor: "var(--border)", background: "var(--warning-bg)" }}
              >
                <p className="text-sm font-medium" style={{ color: "var(--warning-text)" }}>
                  No jobs selected. <button onClick={() => router.push("/jobs")} className="underline">Go to Job Search →</button>
                </p>
              </div>
            )}

            {/* Resume upload */}
            <section
              className="rounded-[var(--radius)] border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
            >
              <h2 className="text-sm font-semibold mb-3" style={{ color: "var(--text-primary)" }}>
                Upload Resumes ({resumes.length}/20)
              </h2>

              {/* Drop zone */}
              <div
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-6 cursor-pointer transition-colors"
                style={{
                  borderColor: dragOver ? "var(--accent)" : "var(--border)",
                  background: dragOver ? "var(--accent-light)" : "var(--bg-page)",
                }}
              >
                {uploading ? (
                  <Loader2 size={24} className="animate-spin" style={{ color: "var(--accent)" }} />
                ) : (
                  <Upload size={24} style={{ color: "var(--text-muted)" }} />
                )}
                <p className="mt-2 text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                  {uploading ? "Extracting text…" : "Drop PDFs here or click to upload"}
                </p>
                <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                  PDF files only · up to 20 resumes
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                multiple
                className="hidden"
                onChange={(e) => Array.from(e.target.files ?? []).forEach(handleFileUpload)}
              />

              {/* Uploaded list */}
              {resumes.length > 0 && (
                <ul className="mt-3 space-y-1">
                  {resumes.map((r) => (
                    <li
                      key={r.name}
                      className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs"
                      style={{ background: "var(--success-bg)" }}
                    >
                      <FileText size={13} style={{ color: "var(--success-text)" }} />
                      <span className="flex-1 truncate font-medium" style={{ color: "var(--success-text)" }}>
                        {r.name}
                      </span>
                      <span style={{ color: "var(--success-text)", opacity: 0.7 }}>
                        {r.resume_text.length.toLocaleString()} chars
                      </span>
                      <button onClick={() => removeResume(r.name)} className="p-0.5 rounded hover:opacity-70">
                        <X size={13} style={{ color: "var(--success-text)" }} />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {/* Error */}
            {error && (
              <div className="rounded-lg p-3 text-sm" style={{ background: "var(--danger-bg)", color: "var(--danger-text)" }}>
                {error}
              </div>
            )}

            {/* Info note */}
            <div
              className="flex items-start gap-2 rounded-lg p-3 text-xs"
              style={{ background: "var(--accent-light)", color: "var(--accent)" }}
            >
              <Info size={14} className="flex-shrink-0 mt-0.5" />
              <span>
                The NLP pipeline runs full semantic + TF-IDF + Word2Vec analysis for each resume × job pair.
                Larger selections may take 30–90 seconds.
              </span>
            </div>

            {/* Run button */}
            <button
              onClick={runMatch}
              disabled={resumes.length === 0 || selectedJobIds.length === 0}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-lg text-sm font-semibold transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ background: "var(--text-primary)", color: "var(--text-on-primary)" }}
            >
              Run Bulk Match ({resumes.length} resume{resumes.length !== 1 ? "s" : ""} × {selectedJobIds.length} job{selectedJobIds.length !== 1 ? "s" : ""})
              <ArrowRight size={16} />
            </button>
          </div>
        </main>
      )}

      {/* ── LOADING PHASE ── */}
      {phase === "loading" && (
        <div className="flex-1 flex flex-col items-center justify-center gap-4">
          <Loader2 size={40} className="animate-spin" style={{ color: "var(--accent)" }} />
          <div className="text-center">
            <p className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              Running NLP pipeline…
            </p>
            <p className="mt-1 text-sm" style={{ color: "var(--text-muted)" }}>
              {resumes.length} resume{resumes.length !== 1 ? "s" : ""} × {selectedJobIds.length} job{selectedJobIds.length !== 1 ? "s" : ""} = {resumes.length * selectedJobIds.length} analyses
            </p>
            <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
              Semantic similarity · TF-IDF · Word2Vec expansion · Entity matching
            </p>
          </div>
        </div>
      )}

      {/* ── RESULTS PHASE ── */}
      {phase === "results" && result && (
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-full">
            {/* Summary header */}
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
                  Leaderboard
                </h1>
                <p className="mt-0.5 text-sm" style={{ color: "var(--text-secondary)" }}>
                  {result.resumes.length} resume{result.resumes.length !== 1 ? "s" : ""}
                  {" · "}
                  {result.jobs.length} job{result.jobs.length !== 1 ? "s" : ""}
                  {" · "}
                  scores are 0–100
                </p>
              </div>
              <div className="flex items-center gap-2">
                <LegendChip label="Strong (≥70)" bg="var(--success-bg)" text="var(--success-text)" />
                <LegendChip label="Partial (40–69)" bg="var(--warning-bg)" text="var(--warning-text)" />
                <LegendChip label="Weak (<40)" bg="var(--danger-bg)" text="var(--danger-text)" />
                <button
                  onClick={resetAndSearch}
                  className="ml-4 text-xs font-semibold px-3 py-1.5 rounded-lg"
                  style={{ background: "var(--accent)", color: "#fff" }}
                >
                  New search
                </button>
              </div>
            </div>

            {/* Matrix table */}
            <div className="overflow-x-auto rounded-[var(--radius)] border" style={{ borderColor: "var(--border)" }}>
              <table className="w-full text-sm border-collapse" style={{ background: "var(--bg-card)" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)", background: "var(--bg-page)" }}>
                    <th
                      className="sticky left-0 px-4 py-3 text-left text-xs font-bold uppercase tracking-wide whitespace-nowrap z-10"
                      style={{ color: "var(--text-muted)", background: "var(--bg-page)", minWidth: "180px" }}
                    >
                      Resume
                    </th>
                    <th
                      className="px-3 py-3 text-center text-xs font-bold uppercase tracking-wide whitespace-nowrap"
                      style={{ color: "var(--text-muted)", background: "var(--bg-page)" }}
                    >
                      Best
                    </th>
                    {result.jobs.map((j) => (
                      <th
                        key={j.id}
                        className="px-2 py-3 text-center text-xs font-medium"
                        style={{ color: "var(--text-secondary)", background: "var(--bg-page)", minWidth: "80px", maxWidth: "100px" }}
                      >
                        <div className="max-w-[90px] truncate text-center mx-auto" title={j.job_title ?? ""}>
                          {j.job_title ?? `#${j.id}`}
                        </div>
                        {j.city && (
                          <div className="text-[10px] font-normal mt-0.5" style={{ color: "var(--text-muted)" }}>
                            {j.city}
                          </div>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, i) => {
                    const { bg: bestBg, text: bestText } = scoreColor(row.best_score);
                    return (
                      <tr
                        key={row.resume_name}
                        style={{ borderBottom: i < result.rows.length - 1 ? "1px solid var(--border)" : undefined }}
                      >
                        {/* Resume name */}
                        <td
                          className="sticky left-0 px-4 py-3 z-10"
                          style={{ background: "var(--bg-card)" }}
                        >
                          <div className="font-medium text-xs truncate max-w-[160px]" style={{ color: "var(--text-primary)" }} title={row.resume_name}>
                            {row.resume_name}
                          </div>
                          {row.best_job_title && (
                            <div className="text-[10px] mt-0.5 truncate max-w-[160px]" style={{ color: "var(--text-muted)" }} title={row.best_job_title}>
                              Best: {row.best_job_title}
                            </div>
                          )}
                        </td>

                        {/* Best score */}
                        <td className="px-3 py-3 text-center">
                          <span
                            className="inline-block min-w-[2.5rem] rounded-md px-2 py-0.5 text-xs font-extrabold"
                            style={{ background: bestBg, color: bestText }}
                          >
                            {row.best_score}
                          </span>
                        </td>

                        {/* Per-job scores */}
                        {result.jobs.map((j) => {
                          const cell = row.cells[String(j.id)];
                          if (!cell) {
                            return (
                              <td key={j.id} className="px-2 py-2 text-center">
                                <span className="text-xs" style={{ color: "var(--text-muted)" }}>—</span>
                              </td>
                            );
                          }
                          return <ScoreBadge key={j.id} score={cell.score} verdict={cell.verdict} />;
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <p className="mt-3 text-xs" style={{ color: "var(--text-muted)" }}>
              Hover a score cell to see the verdict. Rows sorted by best score descending.
            </p>
          </div>
        </main>
      )}
    </div>
  );
}

function LegendChip({ label, bg, text }: { label: string; bg: string; text: string }) {
  return (
    <span
      className="inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full"
      style={{ background: bg, color: text }}
    >
      {label}
    </span>
  );
}
