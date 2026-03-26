"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

interface AccordionProps {
  title: string;
  score?: number;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export function Accordion({ title, score, children, defaultOpen = false }: AccordionProps) {
  const [open, setOpen] = useState(defaultOpen);

  const scoreColor =
    score === undefined
      ? "var(--text-muted)"
      : score >= 70
        ? "var(--success-text)"
        : score >= 40
          ? "var(--warning-text)"
          : "var(--danger-text)";

  const scoreBg =
    score === undefined
      ? "transparent"
      : score >= 70
        ? "var(--success-bg)"
        : score >= 40
          ? "var(--warning-bg)"
          : "var(--danger-bg)";

  return (
    <div className="border rounded-[var(--radius)] overflow-hidden" style={{ borderColor: "var(--border)" }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 transition-colors hover:bg-[var(--bg-hover)]"
      >
        <div className="flex items-center gap-3">
          {score !== undefined && (
            <span
              className="inline-flex items-center justify-center w-10 h-7 text-xs font-bold rounded-md"
              style={{ background: scoreBg, color: scoreColor }}
            >
              {score}
            </span>
          )}
          <span className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>{title}</span>
        </div>
        <ChevronDown
          size={18}
          className="transition-transform duration-200"
          style={{
            color: "var(--text-muted)",
            transform: open ? "rotate(180deg)" : "rotate(0deg)",
          }}
        />
      </button>
      {open && (
        <div className="px-5 pb-4 pt-0 border-t" style={{ borderColor: "var(--border)" }}>
          {children}
        </div>
      )}
    </div>
  );
}
