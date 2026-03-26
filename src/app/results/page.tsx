"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { FileSearch, ArrowLeft, FlaskConical } from "lucide-react";
import { HeroScore } from "@/components/results/hero-score";
import { SectionAccordion } from "@/components/results/section-accordion";
import { NlpDetailsPanel } from "@/components/results/nlp-details-panel";

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
  nlp_details: {
    jd_sections_parsed: Record<string, string[]>;
    resume_sections_parsed: Record<string, string>;
    resume_entities: Record<string, string[]>;
    tfidf_top_keywords: Record<string, { keyword: string; weight: number }[]>;
    similarity_scores: { tfidf_cosine: number; semantic: number };
  };
}

const sectionLabels: Record<string, string> = {
  skills: "Skills Match",
  experience: "Experience Match",
  education: "Education Match",
  preferred: "Preferred Qualifications",
};

export default function ResultsPage() {
  const router = useRouter();
  const [results, setResults] = useState<MatchResults | null>(null);
  const [showNlpDetails, setShowNlpDetails] = useState(false);

  useEffect(() => {
    const stored = sessionStorage.getItem("matchResults");
    if (!stored) {
      router.push("/");
      return;
    }
    setResults(JSON.parse(stored));
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

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header
        className="flex items-center gap-3 px-8 py-4 border-b bg-[var(--bg-card)]"
        style={{ borderColor: "var(--border)" }}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: "var(--text-primary)" }}
        >
          <FileSearch size={16} className="text-white" />
        </div>
        <span className="text-lg font-bold tracking-tight" style={{ color: "var(--text-primary)" }}>
          ResumeMatch
        </span>
        <div className="ml-auto flex items-center gap-3">
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
            onClick={() => router.push("/")}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold text-white"
            style={{ background: "var(--text-primary)" }}
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
            overallScore={results.overall_score}
            verdict={results.verdict}
            summary={results.summary}
            sections={results.sections}
          />

          {/* Section accordions */}
          <div className="space-y-3">
            {Object.entries(results.sections).map(([key, section]) => (
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

          {/* NLP Details panel */}
          {showNlpDetails && (
            <NlpDetailsPanel nlpDetails={results.nlp_details} />
          )}
        </div>
      </main>
    </div>
  );
}
