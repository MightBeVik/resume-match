import { ScoreGauge } from "@/components/ui/score-gauge";
import { ProgressBar } from "@/components/ui/progress-bar";

interface HeroScoreProps {
  overallScore: number;
  verdict: string;
  summary: string;
  sections: Record<string, { score: number }>;
}

const sectionLabels: Record<string, string> = {
  skills: "Skills",
  experience: "Experience",
  education: "Education",
  preferred: "Preferred",
};

export function HeroScore({ overallScore, verdict, summary, sections }: HeroScoreProps) {
  const verdictColor =
    overallScore >= 70 ? "var(--success-text)" : overallScore >= 40 ? "var(--warning-text)" : "var(--danger-text)";
  const verdictBg =
    overallScore >= 70 ? "var(--success-bg)" : overallScore >= 40 ? "var(--warning-bg)" : "var(--danger-bg)";

  return (
    <div
      className="rounded-[var(--radius)] border p-8 text-center"
      style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
    >
      <ScoreGauge score={overallScore} size={220} />

      <div className="mt-4">
        <span
          className="inline-block px-4 py-1.5 rounded-full text-sm font-bold"
          style={{ background: verdictBg, color: verdictColor }}
        >
          {verdict}
        </span>
      </div>

      <p className="mt-4 text-sm max-w-lg mx-auto leading-relaxed" style={{ color: "var(--text-secondary)" }}>
        {summary}
      </p>

      <div className="mt-8 grid grid-cols-2 gap-x-8 gap-y-3 max-w-md mx-auto">
        {Object.entries(sections).map(([key, section]) => (
          <ProgressBar key={key} label={sectionLabels[key] || key} value={section.score} />
        ))}
      </div>
    </div>
  );
}
