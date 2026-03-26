"use client";

interface ScoreGaugeProps {
  score: number;
  size?: number;
}

export function ScoreGauge({ score, size = 220 }: ScoreGaugeProps) {
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const cx = size / 2;
  const cy = size / 2;

  const toRad = (deg: number) => (deg * Math.PI) / 180;

  const arcPath = (startDeg: number, endDeg: number) => {
    const startX = cx + radius * Math.cos(toRad(startDeg));
    const startY = cy - radius * Math.sin(toRad(startDeg));
    const endX = cx + radius * Math.cos(toRad(endDeg));
    const endY = cy - radius * Math.sin(toRad(endDeg));
    const sweep = startDeg > endDeg ? 1 : 0;
    const largeArc = Math.abs(startDeg - endDeg) > 180 ? 1 : 0;
    return `M ${startX} ${startY} A ${radius} ${radius} 0 ${largeArc} ${sweep} ${endX} ${endY}`;
  };

  // Arc goes from 180° (left) to 0° (right)
  const scoreAngle = 180 - (score / 100) * 180;
  const scoreColor = score >= 70 ? "var(--success)" : score >= 40 ? "var(--warning)" : "var(--danger)";

  return (
    <div className="relative flex flex-col items-center" style={{ width: size, height: size * 0.55 }}>
      <svg width={size} height={size * 0.55 + 10} viewBox={`0 0 ${size} ${size * 0.55 + 10}`}>
        {/* Track */}
        <path
          d={arcPath(180, 0)}
          fill="none"
          stroke="var(--gauge-track)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Score arc */}
        {score > 0 && (
          <path
            d={arcPath(180, scoreAngle)}
            fill="none"
            stroke={scoreColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        )}
      </svg>
      <div className="absolute bottom-0 flex flex-col items-center">
        <span className="text-5xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
          {score}
        </span>
        <span className="text-sm" style={{ color: "var(--text-muted)" }}>match score</span>
      </div>
    </div>
  );
}
