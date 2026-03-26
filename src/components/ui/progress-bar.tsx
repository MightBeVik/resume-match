interface ProgressBarProps {
  label: string;
  value: number;
}

export function ProgressBar({ label, value }: ProgressBarProps) {
  const barColor = value >= 70 ? "var(--success)" : value >= 40 ? "var(--warning)" : "var(--danger)";

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span className="font-semibold" style={{ color: "var(--text-primary)" }}>{value}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--gauge-track)" }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${value}%`, background: barColor }}
        />
      </div>
    </div>
  );
}
