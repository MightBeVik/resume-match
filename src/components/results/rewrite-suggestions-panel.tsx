"use client";

import { useState } from "react";
import { Lightbulb, ChevronDown, ChevronUp, Sparkles } from "lucide-react";

interface RewriteSuggestion {
  section: string;
  missing_item: string;
  suggestion: string;
  related_in_resume: string[];
  w2v_expanded: boolean;
}

interface RewriteSuggestionsPanelProps {
  suggestions: RewriteSuggestion[];
}

const sectionLabels: Record<string, string> = {
  skills: "Skills",
  experience: "Experience",
  education: "Education",
  preferred: "Preferred Qualifications",
};

export function RewriteSuggestionsPanel({ suggestions }: RewriteSuggestionsPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!suggestions || suggestions.length === 0) return null;

  const w2vCount = suggestions.filter((s) => s.w2v_expanded).length;

  return (
    <div
      className="rounded-[var(--radius)] border overflow-hidden"
      style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
    >
      {/* Header toggle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-5 py-4 text-left transition-colors hover:bg-[var(--bg-page)]"
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "var(--warning-bg)" }}
          >
            <Lightbulb size={16} style={{ color: "var(--warning-text)" }} />
          </div>
          <div>
            <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Resume Rewrite Suggestions
            </p>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              {suggestions.length} improvement{suggestions.length !== 1 ? "s" : ""} identified
              {w2vCount > 0 && (
                <span className="ml-2 inline-flex items-center gap-1" style={{ color: "var(--accent)" }}>
                  <Sparkles size={10} />
                  {w2vCount} detected via Word2Vec
                </span>
              )}
            </p>
          </div>
        </div>
        {isOpen ? (
          <ChevronUp size={16} style={{ color: "var(--text-muted)" }} />
        ) : (
          <ChevronDown size={16} style={{ color: "var(--text-muted)" }} />
        )}
      </button>

      {/* Suggestions list */}
      {isOpen && (
        <div className="px-5 pb-5 space-y-3 border-t" style={{ borderColor: "var(--border)" }}>
          <p className="pt-4 text-xs" style={{ color: "var(--text-muted)" }}>
            These items were missing from your resume. Adding them — where truthful — will improve your match score.
          </p>

          {suggestions.map((s, i) => (
            <div
              key={i}
              className="rounded-xl border p-4"
              style={{
                borderColor: s.w2v_expanded ? "var(--accent)" : "var(--border)",
                background: "var(--bg-page)",
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  {/* Section badge + missing item */}
                  <div className="flex flex-wrap items-center gap-2 mb-2">
                    <span
                      className="text-xs font-semibold px-2 py-0.5 rounded-md uppercase tracking-wide"
                      style={{ background: "var(--danger-bg)", color: "var(--danger-text)" }}
                    >
                      {sectionLabels[s.section] ?? s.section}
                    </span>
                    {s.w2v_expanded && (
                      <span
                        className="text-xs font-medium px-2 py-0.5 rounded-md flex items-center gap-1"
                        style={{ background: "var(--accent-light)", color: "var(--accent)" }}
                      >
                        <Sparkles size={10} />
                        Word2Vec
                      </span>
                    )}
                  </div>

                  {/* Missing item */}
                  <p className="text-xs font-semibold mb-1" style={{ color: "var(--text-secondary)" }}>
                    Missing: <span style={{ color: "var(--danger-text)" }}>{s.missing_item}</span>
                  </p>

                  {/* Suggestion */}
                  <p className="text-sm leading-relaxed" style={{ color: "var(--text-primary)" }}>
                    {s.suggestion}
                  </p>

                  {/* Related terms found by W2V */}
                  {s.related_in_resume.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      <span className="text-xs" style={{ color: "var(--text-muted)" }}>Related in your resume:</span>
                      {s.related_in_resume.map((r) => (
                        <span
                          key={r}
                          className="text-xs px-2 py-0.5 rounded-md font-medium"
                          style={{ background: "var(--accent-light)", color: "var(--accent)" }}
                        >
                          {r}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
