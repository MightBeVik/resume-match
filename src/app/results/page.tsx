"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { FileSearch, ArrowLeft, FlaskConical, MapPin, Building2 } from "lucide-react";
import { HeroScore } from "@/components/results/hero-score";
import { SectionAccordion } from "@/components/results/section-accordion";
import { NlpDetailsPanel } from "@/components/results/nlp-details-panel";
import { RewriteSuggestionsPanel } from "@/components/results/rewrite-suggestions-panel";
import { ThemeToggle } from "@/components/theme-toggle";

interface MatchResults {
  overall_score: number;
  verdict: string;
  summary: string;
  sections: Record<string, {
    score: number;
    matched: string[];
    partial: string[];
    missing: string[];
  }>;
  rewrite_suggestions: Array<{
    section: string;
    missing_item: string;
    suggestion: string;
    related_in_resume: string[];
    w2v_expanded: boolean;
  }>;
  nlp_details: {
    jd_sections_parsed: Record<string, string[]>;
    resume_sections_parsed: Record<string, string>;
    resume_entities: Record<string, string[]>;
    tfidf_top_keywords: Record<string, { keyword: string; weight: number }[]>;
    similarity_scores: { tfidf_cosine: number; semantic: number };
  };
}

interface RankedJobResult {
  id: number;
  source?: string | null;
  noc_code?: string | null;
  noc_title?: string | null;
  teer?: number | null;
  broad_category?: string | null;
  job_title?: string | null;
  employer_name?: string | null;
  city?: string | null;
  province?: string | null;
  salary?: string | null;
  date_posted?: string | null;
  overall_score: number;
  verdict: string;
  summary: string;
  top_matches: string[];
  top_gaps: string[];
}

interface RankedResultsPayload {
  total_jobs_considered: number;
  returned: number;
  results: RankedJobResult[];
}

type StoredResults =
  | { mode: "single"; payload: MatchResults }
  | {
      mode: "rank";
      payload: RankedResultsPayload;
      filters?: { province?: string; city?: string; jobTitleQuery?: string; limit?: string };
    };

const sectionLabels: Record<string, string> = {
  skills: "Skills Match",
  experience: "Experience Match",
  education: "Education Match",
  preferred: "Preferred Qualifications",
};

export default function ResultsPage() {
  const router = useRouter();
  const [results, setResults] = useState<StoredResults | null>(null);
  const [showNlpDetails, setShowNlpDetails] = useState(false);

  useEffect(() => {
    const stored = sessionStorage.getItem("matchResults");
    if (!stored) {
      router.push("/");
      return;
    }
    const parsed = JSON.parse(stored);
    if (parsed.mode === "single" || parsed.mode === "rank") {
      setResults(parsed);
      return;
    }
    setResults({ mode: "single", payload: parsed });
  }, [router]);

  if (!results) {
    // Skeleton loading state
    return (
      <div className="h-full flex flex-col">
        <header
          className="flex items-center gap-3 px-8 py-4 border-b bg-[var(--bg-card)]"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="w-8 h-8 rounded-lg bg-[var(--gauge-track)] animate-pulse" />
          <div className="w-32 h-5 rounded bg-[var(--gauge-track)] animate-pulse" />
        </header>
        <main className="flex-1 overflow-auto">
          <div className="max-w-3xl mx-auto px-8 py-10 space-y-6">
            {/* Skeleton gauge */}
            <div className="rounded-[var(--radius)] border p-8 flex flex-col items-center" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
              <div className="w-[220px] h-[120px] rounded-t-full bg-[var(--gauge-track)] animate-pulse" />
              <div className="mt-4 w-28 h-7 rounded-full bg-[var(--gauge-track)] animate-pulse" />
              <div className="mt-4 w-64 h-4 rounded bg-[var(--gauge-track)] animate-pulse" />
              <div className="mt-6 grid grid-cols-2 gap-4 w-full max-w-md">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-6 rounded bg-[var(--gauge-track)] animate-pulse" />
                ))}
              </div>
            </div>
            {/* Skeleton accordions */}
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="rounded-[var(--radius)] border p-5" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-7 rounded-md bg-[var(--gauge-track)] animate-pulse" />
                  <div className="w-40 h-5 rounded bg-[var(--gauge-track)] animate-pulse" />
                </div>
              </div>
            ))}
          </div>
        </main>
      </div>
    );
  }

  if (results.mode === "rank") {
    return (
      <div className="h-full flex flex-col">
        <header
          className="flex items-center justify-between px-8 py-4 border-b bg-[var(--bg-card)]"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: "var(--text-primary)" }}
            >
              <FileSearch size={16} style={{ color: "var(--text-on-primary)" }} />
            </div>
            <span className="text-lg font-bold tracking-tight" style={{ color: "var(--text-primary)" }}>
              ResumeMatch
            </span>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={() => router.push("/about")}
              className="text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
              style={{
                color: "var(--accent)",
                background: "var(--accent-light)",
                border: "1px solid var(--accent)",
              }}
            >
              About
            </button>
            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold"
              style={{ background: "var(--text-primary)", color: "var(--text-on-primary)" }}
            >
              <ArrowLeft size={14} />
              New Search
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-auto">
          <div className="max-w-5xl mx-auto px-8 py-10 space-y-6">
            <div className="rounded-[var(--radius)] border p-8" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
              <h1 className="text-3xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
                Ranked Job Matches
              </h1>
              <p className="mt-3 text-sm" style={{ color: "var(--text-secondary)" }}>
                Returned {results.payload.returned} jobs from a candidate pool of {results.payload.total_jobs_considered}
                {results.filters?.province ? ` in ${results.filters.province}` : ""}
                {results.filters?.city ? ` / ${results.filters.city}` : ""}.
              </p>
            </div>

            <div className="space-y-4">
              {results.payload.results.map((job, index) => (
                <div
                  key={job.id}
                  className="rounded-[var(--radius)] border p-6"
                  style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
                >
                  <div className="flex flex-wrap items-start gap-4 justify-between">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.2em]" style={{ color: "var(--text-muted)" }}>
                        Match #{index + 1}
                      </p>
                      <h2 className="mt-1 text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
                        {job.job_title || job.noc_title || "Untitled role"}
                      </h2>
                      <div className="mt-2 flex flex-wrap items-center gap-4 text-sm" style={{ color: "var(--text-secondary)" }}>
                        <span className="inline-flex items-center gap-1.5"><Building2 size={14} />{job.employer_name || "Unknown employer"}</span>
                        <span className="inline-flex items-center gap-1.5"><MapPin size={14} />{[job.city, job.province].filter(Boolean).join(", ") || "Unknown location"}</span>
                        {job.salary && <span>{job.salary}</span>}
                      </div>
                    </div>
                    <div className="min-w-32 rounded-2xl border px-5 py-4 text-center" style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}>
                      <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>Score</p>
                      <p className="mt-1 text-3xl font-black" style={{ color: "var(--text-primary)" }}>{job.overall_score}</p>
                      <p className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>{job.verdict}</p>
                    </div>
                  </div>

                  <p className="mt-4 text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                    {job.summary}
                  </p>

                  <div className="mt-5 grid gap-4 md:grid-cols-2">
                    <div className="rounded-xl border p-4" style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}>
                      <p className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--success-text)" }}>
                        Top Matches
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {job.top_matches.length > 0 ? job.top_matches.map((item) => (
                          <span key={item} className="rounded-md px-2.5 py-1 text-xs font-medium" style={{ background: "var(--success-bg)", color: "var(--success-text)" }}>
                            {item}
                          </span>
                        )) : <span className="text-xs italic" style={{ color: "var(--text-muted)" }}>No strong matches surfaced</span>}
                      </div>
                    </div>
                    <div className="rounded-xl border p-4" style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}>
                      <p className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--danger-text)" }}>
                        Top Gaps
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {job.top_gaps.length > 0 ? job.top_gaps.map((item) => (
                          <span key={item} className="rounded-md px-2.5 py-1 text-xs font-medium" style={{ background: "var(--danger-bg)", color: "var(--danger-text)" }}>
                            {item}
                          </span>
                        )) : <span className="text-xs italic" style={{ color: "var(--text-muted)" }}>No major gaps surfaced</span>}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header
        className="flex items-center justify-between px-8 py-4 border-b bg-[var(--bg-card)]"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "var(--text-primary)" }}
          >
            <FileSearch size={16} style={{ color: "var(--text-on-primary)" }} />
          </div>
          <span className="text-lg font-bold tracking-tight" style={{ color: "var(--text-primary)" }}>
            ResumeMatch
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <ThemeToggle />
          <button
            onClick={() => setShowNlpDetails(!showNlpDetails)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors"
            style={{
              borderColor: showNlpDetails ? "var(--accent)" : "var(--border)",
              color: showNlpDetails ? "var(--accent)" : "var(--text-secondary)",
              background: showNlpDetails ? "var(--accent-light)" : "transparent",
            }}
          >
            <FlaskConical size={14} />
            NLP Details
          </button>
          <button
            onClick={() => router.push("/about")}
            className="text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
            style={{
              color: "var(--accent)",
              background: "var(--accent-light)",
              border: "1px solid var(--accent)",
            }}
          >
            About
          </button>
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold"
            style={{ background: "var(--text-primary)", color: "var(--text-on-primary)" }}
          >
            <ArrowLeft size={14} />
            New Analysis
          </button>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-3xl mx-auto px-8 py-10 space-y-6">
          {/* Hero score */}
          <HeroScore
            overallScore={results.payload.overall_score}
            verdict={results.payload.verdict}
            summary={results.payload.summary}
            sections={results.payload.sections}
          />

          {/* Section accordions */}
          <div className="space-y-3">
            {Object.entries(results.payload.sections).map(([key, section]) => (
              <SectionAccordion
                key={key}
                title={sectionLabels[key] || key}
                score={section.score}
                matched={section.matched}
                partial={section.partial}
                missing={section.missing}
              />
            ))}
          </div>

          {/* Rewrite suggestions powered by Word2Vec */}
          {results.payload.rewrite_suggestions?.length > 0 && (
            <RewriteSuggestionsPanel suggestions={results.payload.rewrite_suggestions} />
          )}

          {/* NLP Details panel */}
          {showNlpDetails && (
            <NlpDetailsPanel nlpDetails={results.payload.nlp_details} />
          )}
        </div>
      </main>
    </div>
  );
}
