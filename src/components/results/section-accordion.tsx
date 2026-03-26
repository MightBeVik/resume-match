import { Accordion } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";

interface SectionAccordionProps {
  title: string;
  score: number;
  matched: string[];
  partial: string[];
  missing: string[];
}

export function SectionAccordion({ title, score, matched, partial, missing }: SectionAccordionProps) {
  const total = matched.length + partial.length + missing.length;

  return (
    <Accordion title={title} score={score}>
      <div className="pt-3 space-y-4">
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          {matched.length} matched · {partial.length} partial · {missing.length} missing · {total} total
        </p>

        {matched.length > 0 && (
          <div>
            <p className="text-xs font-semibold mb-2 uppercase tracking-wide" style={{ color: "var(--success-text)" }}>
              Matched
            </p>
            <div className="flex flex-wrap gap-2">
              {matched.map((item) => (
                <Badge key={item} variant="matched">{item}</Badge>
              ))}
            </div>
          </div>
        )}

        {partial.length > 0 && (
          <div>
            <p className="text-xs font-semibold mb-2 uppercase tracking-wide" style={{ color: "var(--warning-text)" }}>
              Partial Match
            </p>
            <div className="flex flex-wrap gap-2">
              {partial.map((item) => (
                <Badge key={item} variant="partial">{item}</Badge>
              ))}
            </div>
          </div>
        )}

        {missing.length > 0 && (
          <div>
            <p className="text-xs font-semibold mb-2 uppercase tracking-wide" style={{ color: "var(--danger-text)" }}>
              Missing
            </p>
            <div className="flex flex-wrap gap-2">
              {missing.map((item) => (
                <Badge key={item} variant="missing">{item}</Badge>
              ))}
            </div>
          </div>
        )}
      </div>
    </Accordion>
  );
}
