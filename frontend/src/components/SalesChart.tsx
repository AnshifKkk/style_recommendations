import type { QuarterSales } from "../api";
import { quarterLabel } from "../api";

/** Success score per past quarter as bars; hover a bar for the exact values. */
export default function SalesChart({ history }: { history: QuarterSales[] }) {
  const width = 560;
  const height = 180;
  const pad = { top: 16, bottom: 34, left: 32, right: 8 };
  const max = Math.max(1, ...history.map((h) => h.success_score));
  const slot = (width - pad.left - pad.right) / Math.max(history.length, 1);
  const y = (v: number) => pad.top + (height - pad.top - pad.bottom) * (1 - v / max);
  const ticks = [0, max / 2, max];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Success score per quarter">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.left} x2={width - pad.right} y1={y(t)} y2={y(t)} stroke="#e5e5e5" />
          <text x={pad.left - 6} y={y(t) + 4} textAnchor="end" fontSize="10" fill="#737373">
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      {history.map((h, i) => {
        const x = pad.left + i * slot + slot * 0.2;
        const [season, year] = quarterLabel(h.quarter_id).split(" ");
        return (
          <g key={h.quarter_id}>
            <rect x={x} y={y(h.success_score)} width={slot * 0.6} height={y(0) - y(h.success_score)} rx={4} fill="#262626">
              <title>{`${quarterLabel(h.quarter_id)}: score ${h.success_score.toFixed(2)}, ${h.units.toLocaleString()} units, ${h.articles} articles`}</title>
            </rect>
            <text x={x + slot * 0.3} y={height - 18} textAnchor="middle" fontSize="10" fill="#525252">{season}</text>
            <text x={x + slot * 0.3} y={height - 6} textAnchor="middle" fontSize="10" fill="#a3a3a3">{year}</text>
          </g>
        );
      })}
    </svg>
  );
}
