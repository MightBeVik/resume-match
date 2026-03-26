interface BadgeProps {
  variant: "matched" | "partial" | "missing";
  children: React.ReactNode;
}

export function Badge({ variant, children }: BadgeProps) {
  const styles = {
    matched: { background: "var(--success-bg)", color: "var(--success-text)" },
    partial: { background: "var(--warning-bg)", color: "var(--warning-text)" },
    missing: { background: "var(--danger-bg)", color: "var(--danger-text)" },
  };

  return (
    <span
      className="inline-block px-3 py-1 text-xs font-medium rounded-md"
      style={styles[variant]}
    >
      {children}
    </span>
  );
}
