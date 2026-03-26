import { Accordion } from "@/components/ui/accordion";

interface NlpDetailsPanelProps {
  nlpDetails: {
    jd_sections_parsed: Record<string, string[]>;
    resume_sections_parsed: Record<string, string>;
    resume_entities: Record<string, string[]>;
    tfidf_top_keywords: Record<string, { keyword: string; weight: number }[]>;
    similarity_scores: { tfidf_cosine: number; semantic: number };
  };
}

export function NlpDetailsPanel({ nlpDetails }: NlpDetailsPanelProps) {
  const {
    jd_sections_parsed,
    resume_sections_parsed,
    resume_entities,
    tfidf_top_keywords,
    similarity_scores,
  } = nlpDetails;

  return (
    <div
      className="rounded-[var(--radius)] border overflow-hidden"
      style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
    >
      <div
        className="px-5 py-3 border-b"
        style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
      >
        <h3 className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
          NLP Pipeline Details
        </h3>
        <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
          Technical breakdown of how the analysis was computed
        </p>
      </div>

      <div className="p-4 space-y-3">
        {/* Similarity Scores */}
        <Accordion title="Similarity Scores" defaultOpen>
          <div className="pt-3 grid grid-cols-2 gap-4">
            <div className="p-3 rounded-lg" style={{ background: "var(--bg-page)" }}>
              <p className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>TF-IDF Cosine</p>
              <p className="text-2xl font-bold mt-1" style={{ color: "var(--text-primary)" }}>
                {(similarity_scores.tfidf_cosine * 100).toFixed(1)}%
              </p>
            </div>
            <div className="p-3 rounded-lg" style={{ background: "var(--bg-page)" }}>
              <p className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>Semantic</p>
              <p className="text-2xl font-bold mt-1" style={{ color: "var(--text-primary)" }}>
                {(similarity_scores.semantic * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </Accordion>

        {/* TF-IDF Keywords */}
        <Accordion title="TF-IDF Top Keywords">
          <div className="pt-3 grid grid-cols-2 gap-6">
            {Object.entries(tfidf_top_keywords).map(([source, keywords]) => (
              <div key={source}>
                <p className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-muted)" }}>
                  {source === "job_description" ? "Job Description" : "Resume"}
                </p>
                <div className="space-y-1.5">
                  {keywords.slice(0, 10).map((kw) => (
                    <div key={kw.keyword} className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--gauge-track)" }}>
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${Math.min(kw.weight * 200, 100)}%`, background: "var(--accent)" }}
                        />
                      </div>
                      <span className="text-xs w-24 truncate" style={{ color: "var(--text-secondary)" }}>
                        {kw.keyword}
                      </span>
                      <span className="text-xs font-mono w-10 text-right" style={{ color: "var(--text-muted)" }}>
                        {kw.weight.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Accordion>

        {/* Extracted Entities */}
        <Accordion title="Extracted Resume Entities">
          <div className="pt-3 space-y-3">
            {Object.entries(resume_entities).map(([type, entities]) => (
              <div key={type}>
                <p className="text-xs font-semibold uppercase tracking-wide mb-1.5" style={{ color: "var(--text-muted)" }}>
                  {type}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {entities.length > 0 ? entities.map((entity) => (
                    <span
                      key={entity}
                      className="px-2.5 py-0.5 text-xs rounded-md"
                      style={{ background: "var(--accent-light)", color: "var(--accent)" }}
                    >
                      {entity}
                    </span>
                  )) : (
                    <span className="text-xs italic" style={{ color: "var(--text-muted)" }}>None detected</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Accordion>

        {/* JD Sections Parsed */}
        <Accordion title="Job Description Sections Parsed">
          <div className="pt-3 space-y-3">
            {Object.entries(jd_sections_parsed).map(([section, items]) => (
              <div key={section}>
                <p className="text-xs font-semibold uppercase tracking-wide mb-1.5" style={{ color: "var(--text-muted)" }}>
                  {section}
                </p>
                {items.length > 0 ? (
                  <ul className="text-xs space-y-1" style={{ color: "var(--text-secondary)" }}>
                    {items.map((item, i) => (
                      <li key={i} className="pl-3 border-l-2" style={{ borderColor: "var(--border)" }}>
                        {item}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs italic" style={{ color: "var(--text-muted)" }}>No items extracted</p>
                )}
              </div>
            ))}
          </div>
        </Accordion>

        {/* Resume Sections Parsed */}
        <Accordion title="Resume Sections Parsed">
          <div className="pt-3 space-y-3">
            {Object.entries(resume_sections_parsed).map(([section, text]) => (
              <div key={section}>
                <p className="text-xs font-semibold uppercase tracking-wide mb-1.5" style={{ color: "var(--text-muted)" }}>
                  {section}
                </p>
                {text ? (
                  <p className="text-xs leading-relaxed whitespace-pre-line" style={{ color: "var(--text-secondary)" }}>
                    {text.slice(0, 300)}{text.length > 300 ? "..." : ""}
                  </p>
                ) : (
                  <span className="text-xs italic" style={{ color: "var(--text-muted)" }}>Not detected</span>
                )}
              </div>
            ))}
          </div>
        </Accordion>
      </div>
    </div>
  );
}
